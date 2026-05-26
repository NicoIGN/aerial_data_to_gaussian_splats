#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

try:
    import laspy
except ImportError:
    print("Erreur: laspy requis. Installe: pip install laspy[lazrs]")
    sys.exit(1)

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    print("Erreur: scipy requis. Installe: pip install scipy")
    sys.exit(1)


IMAGE_EXTS = {".jp2", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def log(msg: str, level: int, verbose: int):
    if verbose >= level:
        print(msg)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def build_image_index(images_dir: Path):
    index = {}
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        index[p.stem] = p
    return index


def parse_sensor_streaming(xml_path: Path):
    sensor_info = {}

    for event, elem in ET.iterparse(str(xml_path), events=("end",)):
        if elem.tag != "sensor":
            continue

        try:
            usefull_frame = elem.find("usefull-frame/rect")
            focal_pt = elem.find("focal/pt3d")
            pixel_size_txt = elem.findtext("pixel_size")

            if usefull_frame is None or focal_pt is None or pixel_size_txt is None:
                elem.clear()
                continue

            width = int(float(usefull_frame.findtext("w")))
            height = int(float(usefull_frame.findtext("h")))
            cx = float(focal_pt.findtext("x"))
            cy = float(focal_pt.findtext("y"))
            fx = float(focal_pt.findtext("z"))
            fy = fx
            pixel_size_m = float(pixel_size_txt)
            focal_mm = fx * pixel_size_m * 1000.0

            sensor_info = {
                "width": width,
                "height": height,
                "cx": cx,
                "cy": cy,
                "fx": fx,
                "fy": fy,
                "pixel_size_m": pixel_size_m,
                "focal_mm": focal_mm,
                "sensor_name": elem.findtext("name"),
                "serial_number": elem.findtext("serial-number"),
                "objectif": elem.findtext("objectif"),
                "orientation": elem.findtext("orientation"),
            }
            elem.clear()
            break
        finally:
            elem.clear()

    if not sensor_info:
        raise ValueError("Impossible de lire le bloc <sensor> dans le XML.")

    return sensor_info


def iter_cliches_streaming(xml_path: Path, image_index: dict, verbose: int = 1):
    for event, elem in ET.iterparse(str(xml_path), events=("end",)):
        if elem.tag != "cliche":
            continue

        try:
            image_name = elem.findtext("image")
            if not image_name:
                log("[WARN] Cliche sans balise <image>.", 2, verbose)
                continue

            image_name = image_name.strip()
            image_key = Path(image_name).stem

            img_path = image_index.get(image_key)
            if img_path is None:
                log(f"[WARN] Image {image_key} mentionnée dans le XML mais absente du dossier images.", 2, verbose)
                continue

            model = elem.find("model")
            if model is None:
                log(f"[WARN] Balise <model> absente pour {image_key}.", 2, verbose)
                continue

            pt3d = model.find("pt3d")
            quat = model.find("quaternion")
            if pt3d is None or quat is None:
                log(f"[WARN] Pose incomplète pour {image_key}: pt3d/quaternion manquant.", 2, verbose)
                continue

            center = np.array([
                float(pt3d.findtext("x")),
                float(pt3d.findtext("y")),
                float(pt3d.findtext("z")),
            ], dtype=np.float64)

            qx = float(quat.findtext("x"))
            qy = float(quat.findtext("y"))
            qz = float(quat.findtext("z"))
            qw = float(quat.findtext("w"))

            yield {
                "xml_image_name": image_name,
                "image_path": img_path,
                "center": center,
                "quat_xyzw": np.array([qx, qy, qz, qw], dtype=np.float64),
            }

        finally:
            elem.clear()


def pose_xml_to_colmap(quat_xyzw, center_xyz, assume_camera_to_world=True, axis_conv=None):
    rot = R.from_quat(quat_xyzw)

    if assume_camera_to_world:
        rot_cw = rot.inv()
    else:
        rot_cw = rot

    R_cw = rot_cw.as_matrix()

    if axis_conv is not None:
        R_cw = axis_conv @ R_cw

    t = -R_cw @ center_xyz

    rot_final = R.from_matrix(R_cw)
    qx, qy, qz, qw = rot_final.as_quat()
    qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
    return qvec, t, R_cw


def write_cameras_txt(path: Path, width: int, height: int, fx: float, fy: float, cx: float, cy: float):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fx:.12f} {fy:.12f} {cx:.12f} {cy:.12f}\n")


def write_images_txt(path: Path, frames):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(frames)}\n")
        for fr in frames:
            q = fr["qvec"]
            t = fr["tvec"]
            f.write(
                f'{fr["image_id"]} '
                f'{q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f} '
                f'{t[0]:.12f} {t[1]:.12f} {t[2]:.12f} '
                f'1 {fr["frame_name"]}\n'
            )
            f.write("\n")


def write_points3D_txt(path: Path, pts_xyz, pts_rgb=None):
    if pts_rgb is None:
        pts_rgb = np.full((len(pts_xyz), 3), 200, dtype=np.uint8)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write(f"# Number of points: {len(pts_xyz)}\n")
        for i, (p, c) in enumerate(zip(pts_xyz, pts_rgb), start=1):
            f.write(
                f"{i} "
                f"{p[0]:.12f} {p[1]:.12f} {p[2]:.12f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} "
                f"0.0\n"
            )


def read_laz_points(laz_path: Path, stride: int):
    las = laspy.read(str(laz_path))
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)

    rgb = None
    has_rgb = all(hasattr(las, a) for a in ("red", "green", "blue"))
    if has_rgb:
        rgb = np.vstack((las.red, las.green, las.blue)).T
        if rgb.size > 0:
            if rgb.max() > 255:
                rgb = np.clip(rgb / 256.0, 0, 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)

    stride = max(1, int(stride))
    xyz = xyz[::stride]
    if rgb is not None:
        rgb = rgb[::stride]

    return xyz, rgb


def write_ply(path: Path, xyz, rgb=None):
    n = len(xyz)
    if rgb is None:
        rgb = np.full((n, 3), 200, dtype=np.uint8)

    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        for p, c in zip(xyz, rgb):
            f.write(struct.pack("<fffBBB", float(p[0]), float(p[1]), float(p[2]),
                                int(c[0]), int(c[1]), int(c[2])))


def build_transforms_json(path: Path, frames, width, height, fx, fy, cx, cy):
    data = {
        "w": width,
        "h": height,
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "camera_model": "PINHOLE",
        "frames": []
    }

    for fr in frames:
        T_c2w = np.eye(4, dtype=np.float64)
        T_c2w[:3, :3] = fr["R_cw"].T
        T_c2w[:3, 3] = fr["center"]

        data["frames"].append({
            "file_path": f'images/{fr["frame_name"]}',
            "transform_matrix": T_c2w.tolist(),
            "colmap_im_id": fr["image_id"],
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def try_write_colmap_bin(sparse_dir: Path, verbose: int):
    try:
        import pycolmap
    except ImportError:
        log("[INFO] pycolmap non installé: génération .txt uniquement.", 1, verbose)
        return False

    try:
        recon = pycolmap.Reconstruction()
        recon.read_text(str(sparse_dir))
        recon.write_binary(str(sparse_dir))
        log("[INFO] Fichiers .bin générés via pycolmap.", 1, verbose)
        return True
    except Exception as e:
        log(f"[WARN] Impossible de générer les .bin avec pycolmap: {e}", 2, verbose)
        return False


def link_symlink(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def axis_convention_matrix(name: str):
    if name == "identity":
        return np.eye(3, dtype=np.float64)
    if name == "flip_yz":
        return np.diag([1.0, -1.0, -1.0]).astype(np.float64)
    if name == "flip_y":
        return np.diag([1.0, -1.0, 1.0]).astype(np.float64)
    if name == "flip_z":
        return np.diag([1.0, 1.0, -1.0]).astype(np.float64)
    raise ValueError(f"Convention d'axes inconnue: {name}")


def touch_if_missing(path: Path):
    if not path.exists():
        path.touch()


def main():
    ap = argparse.ArgumentParser(
        description="Convertit un XML TA + LAZ + images vers une structure de sortie type COLMAP."
    )
    ap.add_argument("--xml", required=True, help="Fichier XML d'orientation propriétaire")
    ap.add_argument("--laz", required=True, help="Fichier .LAZ")
    ap.add_argument("--images", required=True, help="Dossier des images source")
    ap.add_argument("--out", required=True, help="Dossier de sortie")
    ap.add_argument("--subsample", type=int, default=10, help="Facteur de sous-échantillonnage du LAZ")
    ap.add_argument("--assume-camera-to-world", action="store_true",
                    help="Interprète le quaternion XML comme caméra->monde")
    ap.add_argument("--axis-convention", default="identity",
                    choices=["identity", "flip_yz", "flip_y", "flip_z"],
                    help="Convention fixe appliquée au repère caméra avant export COLMAP")
    ap.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
                    help="0=silencieux, 1=info, 2=warn+info")
    args = ap.parse_args()

    xml_path = Path(args.xml)
    laz_path = Path(args.laz)
    images_dir = Path(args.images)
    out_dir = Path(args.out)

    out_images = out_dir / "images"
    out_colmap = out_dir / "colmap"
    out_sparse = out_colmap / "sparse" / "0"
    out_models_0 = out_sparse / "models" / "0"

    for d in [out_images, out_sparse, out_models_0]:
        ensure_dir(d)

    log("[1/8] Indexation des images existantes...", 1, args.verbose)
    image_index = build_image_index(images_dir)
    log(f"  {len(image_index)} images indexées.", 1, args.verbose)

    if not image_index:
        print("Aucune image compatible trouvée dans le dossier fourni.")
        sys.exit(2)

    log("[2/8] Lecture streaming des paramètres capteur...", 1, args.verbose)
    sensor = parse_sensor_streaming(xml_path)
    width = sensor["width"]
    height = sensor["height"]
    fx = sensor["fx"]
    fy = sensor["fy"]
    cx = sensor["cx"]
    cy = sensor["cy"]

    log(f"  Capteur: {sensor.get('sensor_name')}", 1, args.verbose)
    log(f"  Taille: {width}x{height}", 1, args.verbose)
    log(f"  Focale: {sensor.get('focal_mm'):.3f} mm", 1, args.verbose)
    log(f"  Focale px: ({fx:.3f}, {fy:.3f})", 1, args.verbose)
    log(f"  Point principal: ({cx:.3f}, {cy:.3f})", 1, args.verbose)

    axis_conv = axis_convention_matrix(args.axis_convention)

    log("[3/8] Lecture streaming des clichés valides...", 1, args.verbose)
    frames = []
    num_found = 0

    for idx, item in enumerate(iter_cliches_streaming(xml_path, image_index, verbose=args.verbose)):
        num_found += 1
        frame_name = f"frame_{idx:05d}.png"

        dst_img = out_images / frame_name
        link_symlink(item["image_path"], dst_img)

        qvec, tvec, R_cw = pose_xml_to_colmap(
            item["quat_xyzw"],
            item["center"],
            assume_camera_to_world=args.assume_camera_to_world,
            axis_conv=axis_conv
        )

        frames.append({
            "image_id": idx + 1,
            "frame_name": frame_name,
            "source_image": str(item["image_path"]),
            "xml_image_name": item["xml_image_name"],
            "center": item["center"],
            "qvec": qvec,
            "tvec": tvec,
            "R_cw": R_cw,
        })

        if num_found % 500 == 0:
            log(f"  {num_found} clichés valides traités...", 1, args.verbose)

    log(f"  Total clichés retenus: {len(frames)}", 1, args.verbose)

    if not frames:
        print("Aucun cliché valide trouvé: images absentes ou poses incomplètes.")
        sys.exit(3)

    log("[4/8] Lecture et sous-échantillonnage du LAZ...", 1, args.verbose)
    pts_xyz, pts_rgb = read_laz_points(laz_path, args.subsample)
    log(f"  {len(pts_xyz)} points conservés.", 1, args.verbose)

    log("[5/8] Écriture cameras.txt...", 1, args.verbose)
    write_cameras_txt(out_sparse / "cameras.txt", width, height, fx, fy, cx, cy)

    log("[6/8] Écriture images.txt et points3D.txt...", 1, args.verbose)
    write_images_txt(out_sparse / "images.txt", frames)
    write_points3D_txt(out_sparse / "points3D.txt", pts_xyz, pts_rgb)

    log("[7/8] Export sparse_pc.ply et transforms.json...", 1, args.verbose)
    write_ply(out_dir / "sparse_pc.ply", pts_xyz, pts_rgb)
    build_transforms_json(out_dir / "transforms.json", frames, width, height, fx, fy, cx, cy)

    log("[8/8] Conversion binaire optionnelle...", 1, args.verbose)
    try_write_colmap_bin(out_sparse, args.verbose)

    touch_if_missing(out_sparse / "database.db")
    touch_if_missing(out_sparse / "frames.bin")

    print("\nTerminé.")
    print(f"Sortie: {out_dir}")


if __name__ == "__main__":
    main()

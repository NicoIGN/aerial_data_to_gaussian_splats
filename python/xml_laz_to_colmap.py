#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import struct
import subprocess
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
            orientation_txt = elem.findtext("orientation")

            if usefull_frame is None or focal_pt is None or pixel_size_txt is None:
                elem.clear()
                continue

            width = int(float(usefull_frame.findtext("w")))
            height = int(float(usefull_frame.findtext("h")))

            # ATTENTION:
            # Dans le C++, focal.pt3d.x/y sont cPPA/lPPA, pas forcément un cx/cy COLMAP direct.
            cppa = float(focal_pt.findtext("x"))
            lppa = float(focal_pt.findtext("y"))
            focal_px = float(focal_pt.findtext("z"))

            pixel_size_m = float(pixel_size_txt)
            focal_mm = focal_px * pixel_size_m * 1000.0

            sensor_info = {
                "width": width,
                "height": height,
                "cppa": cppa,
                "lppa": lppa,
                "fx": focal_px,
                "fy": focal_px,
                "pixel_size_m": pixel_size_m,
                "focal_mm": focal_mm,
                "sensor_name": elem.findtext("name"),
                "serial_number": elem.findtext("serial-number"),
                "objectif": elem.findtext("objectif"),
                "orientation": int(orientation_txt) if orientation_txt is not None else 0,
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


def pose_xml_to_colmap_cpp_exact(quat_xyzw, center_xyz):
    """
    Reproduit exactement la logique C++ montrée dans Shot::initialize() :

        quat.getRotation(mat)

        rotation[0][0] =  mat(0, 1)
        rotation[0][1] =  mat(1, 1)
        rotation[0][2] =  mat(2, 1)
        rotation[1][0] =  mat(0, 0)
        rotation[1][1] =  mat(1, 0)
        rotation[1][2] =  mat(2, 0)
        rotation[2][0] = -mat(0, 2)
        rotation[2][1] = -mat(1, 2)
        rotation[2][2] = -mat(2, 2)

    Puis projection:
        Xc = R_cw * (Xw - C)
        t  = -R_cw * C
    """
    rot = R.from_quat(quat_xyzw)
    mat = rot.as_matrix()

    R_cw = np.array([
        [ mat[0, 1],  mat[1, 1],  mat[2, 1]],
        [ mat[0, 0],  mat[1, 0],  mat[2, 0]],
        [-mat[0, 2], -mat[1, 2], -mat[2, 2]],
    ], dtype=np.float64)

    det = np.linalg.det(R_cw)
    if det <= 0:
        raise ValueError(f"Matrice rotation invalide pour COLMAP (det={det})")

    t = -R_cw @ center_xyz

    rot_final = R.from_matrix(R_cw)
    qx, qy, qz, qw = rot_final.as_quat()
    qvec = np.array([qw, qx, qy, qz], dtype=np.float64)

    return qvec, t, R_cw


def principal_point_from_ppa(width, height, cppa, lppa, ppa_mode):
    """
    Convertit cPPA/lPPA MATIS vers un principal point pixel supposé.

    INCERTAIN : faute d'avoir le code exact de BundleToImage côté intrinseque,
    on expose plusieurs hypothèses.
    """
    if ppa_mode == "direct":
        # Hypothèse la plus naïve
        cx = cppa
        cy = lppa
    elif ppa_mode == "center_plus":
        # Hypothèse fréquente si PPA est exprimé autour du centre image
        cx = (width / 2.0) + cppa
        cy = (height / 2.0) + lppa
    elif ppa_mode == "center_minus":
        cx = (width / 2.0) - cppa
        cy = (height / 2.0) - lppa
    elif ppa_mode == "half_pixel_center_plus":
        cx = ((width - 1) / 2.0) + cppa
        cy = ((height - 1) / 2.0) + lppa
    elif ppa_mode == "half_pixel_center_minus":
        cx = ((width - 1) / 2.0) - cppa
        cy = ((height - 1) / 2.0) - lppa
    else:
        raise ValueError(f"ppa_mode inconnu: {ppa_mode}")

    return cx, cy


def apply_sensor_orientation_to_intrinsics(width, height, fx, fy, cx, cy, orientation):
    """
    Applique une rotation capteur sur les intrinsics SI les images exportées
    sont effectivement tournées dans le raster final.

    orientation XML:
      0 -> 0°
      1 -> 180°
      2 -> 90°
      3 -> 270°
    """
    orientation = int(orientation)

    if orientation == 0:
        return width, height, fx, fy, cx, cy

    if orientation == 1:  # 180°
        return width, height, fx, fy, (width - 1 - cx), (height - 1 - cy)

    if orientation == 2:  # 90°
        return height, width, fy, fx, cy, (width - 1 - cx)

    if orientation == 3:  # 270°
        return height, width, fy, fx, (height - 1 - cy), cx

    raise ValueError(f"Orientation capteur inconnue: {orientation}")


def scale_intrinsics(width, height, fx, fy, cx, cy, factor):
    factor = max(float(factor), 1.0)
    width2 = max(1, int(round(width / factor)))
    height2 = max(1, int(round(height / factor)))
    fx2 = fx / factor
    fy2 = fy / factor
    cx2 = cx / factor
    cy2 = cy / factor
    return width2, height2, fx2, fy2, cx2, cy2


def write_cameras_txt(path: Path, width: int, height: int, fx: float, fy: float, cx: float, cy: float):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fx:.12f} {fy:.12f} {cx:.12f} {cy:.12f}\n")


def write_images_txt(path: Path, frames, observations_by_image):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(frames)}\n")

        for fr in frames:
            q = fr["qvec"]
            t = fr["tvec"]
            image_id = fr["image_id"]

            f.write(
                f'{image_id} '
                f'{q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f} '
                f'{t[0]:.12f} {t[1]:.12f} {t[2]:.12f} '
                f'1 {fr["frame_name"]}\n'
            )

            obs = observations_by_image.get(image_id, [])
            line = []
            for o in obs:
                x, y = o["xy"]
                pid = o["point3d_id"]
                line.append(f"{x:.6f} {y:.6f} {pid}")

            f.write(" ".join(line) + "\n")


def write_points3D_txt(path: Path, pts_xyz, pts_rgb=None, tracks_by_point=None):
    if pts_rgb is None:
        pts_rgb = np.full((len(pts_xyz), 3), 200, dtype=np.uint8)

    if tracks_by_point is None:
        tracks_by_point = [[] for _ in range(len(pts_xyz))]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write(f"# Number of points: {len(pts_xyz)}\n")

        for i, (p, c, track) in enumerate(zip(pts_xyz, pts_rgb, tracks_by_point), start=1):
            parts = [
                str(i),
                f"{p[0]:.12f}",
                f"{p[1]:.12f}",
                f"{p[2]:.12f}",
                str(int(c[0])),
                str(int(c[1])),
                str(int(c[2])),
                "0.0",
            ]

            for tr in track:
                parts.append(str(tr["image_id"]))
                parts.append(str(tr["point2d_idx"]))

            f.write(" ".join(parts) + "\n")


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


def project_point(R_cw, t_cw, fx, fy, cx, cy, xyz):
    Xc = R_cw @ xyz + t_cw
    z = float(Xc[2])

    if z <= 1e-9:
        return None

    u = fx * (Xc[0] / z) + cx
    v = fy * (Xc[1] / z) + cy
    return np.array([u, v], dtype=np.float64), z


def build_synthetic_observations(frames, pts_xyz, width, height, fx, fy, cx, cy,
                                 max_points_per_image=None, verbose=1):
    observations_by_image = {fr["image_id"]: [] for fr in frames}
    tracks_by_point = [[] for _ in range(len(pts_xyz))]

    for fr in frames:
        image_id = fr["image_id"]
        R_cw = fr["R_cw"]
        t_cw = fr["tvec"]

        candidates = []

        for pid, xyz in enumerate(pts_xyz, start=1):
            proj = project_point(R_cw, t_cw, fx, fy, cx, cy, xyz)
            if proj is None:
                continue

            uv, depth = proj
            u, v = uv

            if not (0.0 <= u < width and 0.0 <= v < height):
                continue

            candidates.append({
                "xy": uv,
                "depth": depth,
                "point3d_id": pid,
            })

        candidates.sort(key=lambda o: o["depth"])

        if max_points_per_image is not None:
            candidates = candidates[:max_points_per_image]

        for obs in candidates:
            point2d_idx = len(observations_by_image[image_id])

            observations_by_image[image_id].append({
                "xy": obs["xy"],
                "point3d_id": obs["point3d_id"],
            })

            tracks_by_point[obs["point3d_id"] - 1].append({
                "image_id": image_id,
                "point2d_idx": point2d_idx,
            })

        log(f"  Image {image_id}: {len(candidates)} observations synthétiques", 1, verbose)

    return observations_by_image, tracks_by_point


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


def run_decompression_script(script_path: Path, input_dir: Path, output_dir: Path,
                             factor: float, jpeg_quality: int, verbose: int):
    cmd = [
        sys.executable,
        str(script_path),
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--factor", str(factor),
        "--jpg",
        "--jpeg-quality", str(jpeg_quality),
    ]
    if verbose >= 1:
        cmd.append("--verbose")

    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(
        description="Convertit un XML TA + LAZ + images vers une structure de sortie type COLMAP."
    )
    ap.add_argument("--xml", required=True, help="Fichier XML d'orientation propriétaire")
    ap.add_argument("--laz", required=True, help="Fichier .LAZ")
    ap.add_argument("--images", required=True, help="Dossier des images source")
    ap.add_argument("--out", required=True, help="Dossier de sortie")
    ap.add_argument("--subsample", type=int, default=10, help="Facteur de sous-échantillonnage du LAZ")
    ap.add_argument("--image-factor", type=float, default=1.0,
                    help="Facteur de sous-échantillonnage des images (2 = largeur/hauteur divisées par 2)")
    ap.add_argument("--jpeg-quality", type=int, default=95, help="Qualité JPEG de sortie")
    ap.add_argument("--max-observations-per-image", type=int, default=5000,
                    help="Nombre max d'observations synthétiques par image")
    # Hypothèses encore incertaines
    ap.add_argument(
        "--ppa-mode",
        default="center_plus",
        choices=[
            "direct",
            "center_plus",
            "center_minus",
            "half_pixel_center_plus",
            "half_pixel_center_minus",
        ],
        help="Interprétation de cPPA/lPPA en principal point pixel"
    )
    ap.add_argument(
        "--apply-sensor-orientation-to-intrinsics",
        action="store_true",
        help="Applique la rotation capteur XML aux intrinsics exportées"
    )

    ap.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
                    help="0=silencieux, 1=info, 2=warn+info")
    args = ap.parse_args()

    xml_path = Path(args.xml)
    laz_path = Path(args.laz)
    images_dir = Path(args.images)
    out_dir = Path(args.out)

    script_dir = Path(__file__).resolve().parent
    decompression_script = script_dir / "decompress-jp2.py"

    if not decompression_script.exists():
        raise FileNotFoundError(f"Script introuvable: {decompression_script}")

    out_images = out_dir / "colmap" / "images"
    out_colmap = out_dir / "colmap"
    out_sparse = out_colmap / "sparse" / "0"
    out_models_0 = out_sparse / "models" / "0"

    for d in [out_images, out_sparse, out_models_0]:
        ensure_dir(d)

    log("[1/9] Indexation des images existantes...", 1, args.verbose)
    image_index = build_image_index(images_dir)
    log(f"  {len(image_index)} images indexées.", 1, args.verbose)

    if not image_index:
        print("Aucune image compatible trouvée dans le dossier fourni.")
        sys.exit(2)

    log("[2/9] Lecture streaming des paramètres capteur...", 1, args.verbose)
    sensor = parse_sensor_streaming(xml_path)

    width0 = sensor["width"]
    height0 = sensor["height"]
    fx0 = sensor["fx"]
    fy0 = sensor["fy"]
    cppa0 = sensor["cppa"]
    lppa0 = sensor["lppa"]
    orientation0 = int(sensor.get("orientation", 0))

    cx0, cy0 = principal_point_from_ppa(
        width=width0,
        height=height0,
        cppa=cppa0,
        lppa=lppa0,
        ppa_mode=args.ppa_mode,
    )

    width1, height1, fx1, fy1, cx1, cy1 = width0, height0, fx0, fy0, cx0, cy0
    if args.apply_sensor_orientation_to_intrinsics:
        width1, height1, fx1, fy1, cx1, cy1 = apply_sensor_orientation_to_intrinsics(
            width=width0,
            height=height0,
            fx=fx0,
            fy=fy0,
            cx=cx0,
            cy=cy0,
            orientation=orientation0,
        )

    factor = max(float(args.image_factor), 1.0)
    width, height, fx, fy, cx, cy = scale_intrinsics(
        width1, height1, fx1, fy1, cx1, cy1, factor
    )

    log(f"  Capteur: {sensor.get('sensor_name')}", 1, args.verbose)
    log(f"  Orientation capteur XML: {orientation0}", 1, args.verbose)
    log(f"  Taille native: {width0}x{height0}", 1, args.verbose)
    log(f"  Focale px native: ({fx0:.3f}, {fy0:.3f})", 1, args.verbose)
    log(f"  cPPA/lPPA natifs: ({cppa0:.3f}, {lppa0:.3f})", 1, args.verbose)
    log(f"  Principal point avant orientation: ({cx0:.3f}, {cy0:.3f})", 1, args.verbose)
    log(f"  Principal point après orientation: ({cx1:.3f}, {cy1:.3f})", 1, args.verbose)
    log(f"  Taille avant scale: {width1}x{height1}", 1, args.verbose)
    log(f"  Taille exportée: {width}x{height}", 1, args.verbose)
    log(f"  Focale px exportée: ({fx:.3f}, {fy:.3f})", 1, args.verbose)
    log(f"  Principal point exporté: ({cx:.3f}, {cy:.3f})", 1, args.verbose)
    log(f"  ppa_mode: {args.ppa_mode}", 1, args.verbose)
    log(f"  apply_sensor_orientation_to_intrinsics: {args.apply_sensor_orientation_to_intrinsics}", 1, args.verbose)

    log("[3/9] Décompression + sous-échantillonnage des images...", 1, args.verbose)
    run_decompression_script(
        script_path=decompression_script,
        input_dir=images_dir,
        output_dir=out_images,
        factor=factor,
        jpeg_quality=args.jpeg_quality,
        verbose=args.verbose,
    )

    decompressed_index = build_image_index(out_images)
    log(f"  {len(decompressed_index)} images exportées indexées.", 1, args.verbose)

    log("[4/9] Lecture streaming des clichés valides...", 1, args.verbose)
    frames = []
    num_found = 0

    for idx, item in enumerate(iter_cliches_streaming(xml_path, image_index, verbose=args.verbose)):
        num_found += 1

        src_stem = item["image_path"].stem
        exported_img = decompressed_index.get(src_stem)
        if exported_img is None:
            log(f"[WARN] Image exportée absente après décompression: {src_stem}", 2, args.verbose)
            continue

        frame_name = exported_img.name

        qvec, tvec, R_cw = pose_xml_to_colmap_cpp_exact(
            item["quat_xyzw"],
            item["center"],
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

    log("[5/9] Lecture et sous-échantillonnage du LAZ...", 1, args.verbose)
    pts_xyz, pts_rgb = read_laz_points(laz_path, args.subsample)
    log(f"  {len(pts_xyz)} points conservés.", 1, args.verbose)

    log("[6/9] Génération des observations synthétiques...", 1, args.verbose)
    observations_by_image, tracks_by_point = build_synthetic_observations(
        frames=frames,
        pts_xyz=pts_xyz,
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        max_points_per_image=args.max_observations_per_image,
        verbose=args.verbose,
    )

    log("[6/9] Écriture cameras.txt...", 1, args.verbose)
    write_cameras_txt(out_sparse / "cameras.txt", width, height, fx, fy, cx, cy)

    log("[7/9] Écriture images.txt et points3D.txt...", 1, args.verbose)
    write_images_txt(out_sparse / "images.txt", frames, observations_by_image)
    write_points3D_txt(out_sparse / "points3D.txt", pts_xyz, pts_rgb, tracks_by_point)

    log("[8/9] Export transforms.json...", 1, args.verbose)
    build_transforms_json(out_colmap / "transforms.json", frames, width, height, fx, fy, cx, cy)

    log("Conversion au format binaire...", 1, args.verbose)
    try_write_colmap_bin(out_sparse, args.verbose)

    print("\nTerminé.")
    print(f"Sortie: {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
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


def apply_cylindrical_systematism_local_to_image(c, l, transfo2d):
    if transfo2d is None:
        return c, l

    tr_type = transfo2d.get("Type")
    if tr_type != "systematismeCylindriqueTopAero":
        return c, l

    C0 = float(transfo2d.get("C0", 0.0))
    L0 = float(transfo2d.get("L0", 0.0))  # parsé pour complétude, non utilisé dans la formule C++
    S1 = float(transfo2d.get("S1", 0.0))
    S2 = float(transfo2d.get("S2", 0.0))

    ci = c
    li = l

    l = li + (ci - C0) * S1
    c = ci + (ci - C0) * S2

    return c, l


def apply_cylindrical_systematism_image_to_local(c, l, transfo2d):
    if transfo2d is None:
        return c, l

    tr_type = transfo2d.get("Type")
    if tr_type != "systematismeCylindriqueTopAero":
        return c, l

    C0 = float(transfo2d.get("C0", 0.0))
    L0 = float(transfo2d.get("L0", 0.0))  # parsé pour complétude, non utilisé dans la formule C++
    S1 = float(transfo2d.get("S1", 0.0))
    S2 = float(transfo2d.get("S2", 0.0))

    ci = c
    li = l

    l = li - (ci - C0) * S1
    c = ci - (ci - C0) * S2

    return c, l


def parse_con_orientation(con_path: Path):
    tree = ET.parse(con_path)
    root = tree.getroot()

    geometry = root.find("geometry")
    if geometry is None:
        raise ValueError(f"{con_path}: balise <geometry> absente")

    extr = geometry.find("extrinseque")
    intr = geometry.find("intrinseque")
    if extr is None or intr is None:
        raise ValueError(f"{con_path}: balises extrinseque/intrinseque absentes")

    sensor = intr.find("sensor")
    if sensor is None:
        raise ValueError(f"{con_path}: balise intrinseque/sensor absente")

    eucl = extr.find("systeme/euclidien")
    sommet = extr.find("sommet")
    rotation = extr.find("rotation")

    if eucl is None or sommet is None or rotation is None:
        raise ValueError(f"{con_path}: extrinsèque incomplet")

    eucl_x = float(eucl.findtext("x"))
    eucl_y = float(eucl.findtext("y"))

    easting = float(sommet.findtext("easting"))
    northing = float(sommet.findtext("northing"))
    altitude = float(sommet.findtext("altitude"))

    center = np.array([
        eucl_x + easting,
        eucl_y + northing,
        altitude,
    ], dtype=np.float64)

    image2ground_txt = rotation.findtext("Image2Ground")
    image2ground = str(image2ground_txt).strip().lower() == "true"

    def _read_row(row_tag):
        row = rotation.find(f"mat3d/{row_tag}/pt3d")
        if row is None:
            raise ValueError(f"{con_path}: ligne {row_tag} absente dans rotation/mat3d")
        return [
            float(row.findtext("x")),
            float(row.findtext("y")),
            float(row.findtext("z")),
        ]

    M = np.array([
        _read_row("l1"),
        _read_row("l2"),
        _read_row("l3"),
    ], dtype=np.float64)

    if image2ground:
        R_cw = M.T
    else:
        R_cw = M

    det = np.linalg.det(R_cw)
    if det <= 0:
        raise ValueError(f"{con_path}: rotation invalide, det={det}")

    tvec = -R_cw @ center

    rot_final = R.from_matrix(R_cw)
    qx, qy, qz, qw = rot_final.as_quat()
    qvec = np.array([qw, qx, qy, qz], dtype=np.float64)

    width = int(sensor.findtext("image_size/width"))
    height = int(sensor.findtext("image_size/height"))
    cx = float(sensor.findtext("ppa/c"))
    cy = float(sensor.findtext("ppa/l"))
    fx = float(sensor.findtext("ppa/focale"))
    fy = fx

    pixel_size_txt = sensor.findtext("pixel_size")
    pixel_size = float(pixel_size_txt) if pixel_size_txt is not None else None

    transfo2d = None
    tr = sensor.find("transfo2d/tr2delem")
    if tr is not None:
        transfo2d = {
            "Type": tr.attrib.get("Type"),
            "isinterne": tr.attrib.get("isinterne"),
            "C0": float(tr.attrib.get("C0", "0")),
            "L0": float(tr.attrib.get("L0", "0")),
            "S1": float(tr.attrib.get("S1", "0")),
            "S2": float(tr.attrib.get("S2", "0")),
        }

    return {
        "center": center,
        "R_cw": R_cw,
        "tvec": tvec,
        "qvec": qvec,
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "pixel_size": pixel_size,
        "image2ground": image2ground,
        "transfo2d": transfo2d,
    }


def scale_intrinsics(width, height, fx, fy, cx, cy, factor):
    factor = max(float(factor), 1.0)
    width2 = max(1, int(round(width / factor)))
    height2 = max(1, int(round(height / factor)))
    fx2 = fx / factor
    fy2 = fy / factor
    cx2 = cx / factor
    cy2 = cy / factor
    return width2, height2, fx2, fy2, cx2, cy2


def scale_transfo2d(transfo2d, factor):
    if transfo2d is None:
        return None

    factor = max(float(factor), 1.0)

    scaled = dict(transfo2d)
    scaled["C0"] = float(transfo2d.get("C0", 0.0)) / factor
    scaled["L0"] = float(transfo2d.get("L0", 0.0)) / factor

    # S1/S2 sont des pentes en pixel/pixel dans cette formule affine simple le long des colonnes.
    # Ils restent donc inchangés sous simple homothétie isotrope.
    scaled["S1"] = float(transfo2d.get("S1", 0.0))
    scaled["S2"] = float(transfo2d.get("S2", 0.0))

    return scaled


def project_point(R_cw, t_cw, fx, fy, cx, cy, xyz, transfo2d=None):
    Xc = R_cw @ xyz + t_cw
    z = float(Xc[2])

    if z <= 1e-9:
        return None

    u = fx * (Xc[0] / z) + cx
    v = fy * (Xc[1] / z) + cy

    u, v = apply_cylindrical_systematism_local_to_image(u, v, transfo2d)

    return np.array([u, v], dtype=np.float64), z


def build_synthetic_observations(frames, pts_xyz, max_points_per_image=None, verbose=1):
    observations_by_image = {fr["image_id"]: [] for fr in frames}
    tracks_by_point = [[] for _ in range(len(pts_xyz))]

    for fr in frames:
        image_id = fr["image_id"]
        R_cw = fr["R_cw"]
        t_cw = fr["tvec"]
        width = fr["width"]
        height = fr["height"]
        fx = fr["fx"]
        fy = fr["fy"]
        cx = fr["cx"]
        cy = fr["cy"]
        transfo2d = fr.get("transfo2d")

        candidates = []

        for pid, xyz in enumerate(pts_xyz, start=1):
            proj = project_point(R_cw, t_cw, fx, fy, cx, cy, xyz, transfo2d=transfo2d)
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


def write_cameras_txt_single_camera(path: Path, width: int, height: int, fx: float, fy: float, cx: float, cy: float):
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
            camera_id = fr["camera_id"]

            f.write(
                f'{image_id} '
                f'{q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f} '
                f'{t[0]:.12f} {t[1]:.12f} {t[2]:.12f} '
                f'{camera_id} {fr["frame_name"]}\n'
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


def build_transforms_json(path: Path, frames):
    if not frames:
        raise ValueError("Aucune frame pour transforms.json")

    first = frames[0]
    data = {
        "w": first["width"],
        "h": first["height"],
        "fl_x": first["fx"],
        "fl_y": first["fy"],
        "cx": first["cx"],
        "cy": first["cy"],
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


def main():
    ap = argparse.ArgumentParser(
        description="Convertit un dossier images + fichiers .CON + LAZ vers une structure de sortie type COLMAP."
    )
    ap.add_argument("--laz", required=True, help="Fichier .LAZ")
    ap.add_argument("--images", required=True, help="Dossier des images source")
    ap.add_argument("--out", required=True, help="Dossier de sortie")
    ap.add_argument("--subsample", type=int, default=10, help="Facteur de sous-échantillonnage du LAZ")
    ap.add_argument("--image-factor", type=float, default=1.0,
                    help="Facteur de sous-échantillonnage des images (2 = largeur/hauteur divisées par 2)")
    ap.add_argument("--jpeg-quality", type=int, default=95, help="Qualité JPEG de sortie")
    ap.add_argument("--max-observations-per-image", type=int, default=5000,
                    help="Nombre max d'observations synthétiques par image")
    ap.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
                    help="0=silencieux, 1=info, 2=warn+info")
    args = ap.parse_args()

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

    log("[1/7] Indexation des images existantes...", 1, args.verbose)
    image_index = build_image_index(images_dir)
    log(f"  {len(image_index)} images indexées.", 1, args.verbose)

    if not image_index:
        print("Aucune image compatible trouvée dans le dossier fourni.")
        sys.exit(2)

    factor = max(float(args.image_factor), 1.0)

    log("[2/7] Décompression + sous-échantillonnage des images...", 1, args.verbose)
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

    log("[3/7] Lecture des fichiers .CON...", 1, args.verbose)
    frames = []
    intrinsics_ref = None
    skipped_no_con = 0

    for stem, src_img in sorted(image_index.items()):
        con_path = src_img.with_suffix(".CON")
        if not con_path.exists():
            con_path = src_img.with_suffix(".con")

        if not con_path.exists():
            log(f"[WARN] .CON introuvable pour {src_img.name}", 2, args.verbose)
            skipped_no_con += 1
            continue

        exported_img = decompressed_index.get(stem)
        if exported_img is None:
            log(f"[WARN] Image exportée absente après décompression: {stem}", 2, args.verbose)
            continue

        try:
            ori = parse_con_orientation(con_path)
        except Exception as e:
            log(f"[WARN] Impossible de lire {con_path.name}: {e}", 2, args.verbose)
            continue

        width, height, fx, fy, cx, cy = scale_intrinsics(
            ori["width"], ori["height"], ori["fx"], ori["fy"], ori["cx"], ori["cy"], factor
        )
        transfo2d_scaled = scale_transfo2d(ori["transfo2d"], factor)

        if intrinsics_ref is None:
            intrinsics_ref = (width, height, fx, fy, cx, cy)
            log(f"  Intrinsics de référence: {intrinsics_ref}", 1, args.verbose)
        else:
            ref = intrinsics_ref
            cur = (width, height, fx, fy, cx, cy)
            same = (
                ref[0] == cur[0] and
                ref[1] == cur[1] and
                abs(ref[2] - cur[2]) < 1e-6 and
                abs(ref[3] - cur[3]) < 1e-6 and
                abs(ref[4] - cur[4]) < 1e-6 and
                abs(ref[5] - cur[5]) < 1e-6
            )
            if not same:
                raise ValueError(
                    "Le script actuel n'exporte qu'une seule caméra COLMAP. "
                    f"Intrinsics différentes détectées pour {src_img.name}: "
                    f"ref={ref}, cur={cur}"
                )

        frames.append({
            "image_id": len(frames) + 1,
            "camera_id": 1,
            "frame_name": exported_img.name,
            "source_image": str(src_img),
            "con_path": str(con_path),
            "center": ori["center"],
            "qvec": ori["qvec"],
            "tvec": ori["tvec"],
            "R_cw": ori["R_cw"],
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "transfo2d": transfo2d_scaled,
        })

        if transfo2d_scaled is not None:
            log(
                f"  {con_path.name}: transfo2d appliqué = {transfo2d_scaled}",
                1,
                args.verbose,
            )

    log(f"  Total images avec .CON retenues: {len(frames)}", 1, args.verbose)
    if skipped_no_con > 0:
        log(f"  Images ignorées faute de .CON: {skipped_no_con}", 1, args.verbose)

    if not frames:
        print("Aucune image exploitable avec fichier .CON trouvé.")
        sys.exit(3)

    width, height, fx, fy, cx, cy = intrinsics_ref

    log("[4/7] Lecture et sous-échantillonnage du LAZ...", 1, args.verbose)
    pts_xyz, pts_rgb = read_laz_points(laz_path, args.subsample)
    log(f"  {len(pts_xyz)} points conservés.", 1, args.verbose)

    log("[5/7] Génération des observations synthétiques...", 1, args.verbose)
    observations_by_image, tracks_by_point = build_synthetic_observations(
        frames=frames,
        pts_xyz=pts_xyz,
        max_points_per_image=args.max_observations_per_image,
        verbose=args.verbose,
    )

    log("[6/7] Écriture cameras.txt, images.txt, points3D.txt...", 1, args.verbose)
    write_cameras_txt_single_camera(out_sparse / "cameras.txt", width, height, fx, fy, cx, cy)
    write_images_txt(out_sparse / "images.txt", frames, observations_by_image)
    write_points3D_txt(out_sparse / "points3D.txt", pts_xyz, pts_rgb, tracks_by_point)

    log("[7/7] Export transforms.json + conversion binaire...", 1, args.verbose)
    build_transforms_json(out_colmap / "transforms.json", frames)
    try_write_colmap_bin(out_sparse, args.verbose)

    print("\nTerminé.")
    print(f"Sortie: {out_dir}")


if __name__ == "__main__":
    main()

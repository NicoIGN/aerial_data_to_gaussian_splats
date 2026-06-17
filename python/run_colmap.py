#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np


def info(msg: str):
    print(f"[INFO] {msg}")


def warn(msg: str):
    print(f"[WARN] {msg}")


def run_cmd(cmd):
    info("CMD: " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Commande échouée ({result.returncode}): {' '.join(map(str, cmd))}")


def find_colmap_binary(user_value=None):
    if user_value:
        return user_value
    colmap_bin = shutil.which("colmap")
    if colmap_bin is None:
        raise FileNotFoundError(
            "Impossible de trouver le binaire 'colmap'. "
            "Installez COLMAP ou utilisez --colmap /chemin/vers/colmap"
        )
    return colmap_bin


def qvec_to_rotmat(qvec):
    qvec = np.asarray(qvec, dtype=np.float64)
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z,     2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z,     1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y,     2 * y * z + 2 * w * x,     1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def load_colmap_cameras(model_dir: Path):
    bin_path = model_dir / "cameras.bin"
    txt_path = model_dir / "cameras.txt"

    if bin_path.exists():
        return read_cameras_bin(bin_path)
    if txt_path.exists():
        return read_cameras_txt(txt_path)

    raise FileNotFoundError(f"cameras.bin/cameras.txt introuvable dans {model_dir}")


def load_colmap_images(model_dir: Path):
    bin_path = model_dir / "images.bin"
    txt_path = model_dir / "images.txt"

    if bin_path.exists():
        return read_images_bin(bin_path)
    if txt_path.exists():
        return read_images_txt(txt_path)

    raise FileNotFoundError(f"images.bin/images.txt introuvable dans {model_dir}")


def load_colmap_points3D(model_dir: Path):
    bin_path = model_dir / "points3D.bin"
    txt_path = model_dir / "points3D.txt"

    if bin_path.exists():
        return read_points3d_bin(bin_path)
    if txt_path.exists():
        return read_points3d_txt(txt_path)

    raise FileNotFoundError(f"points3D.bin/points3D.txt introuvable dans {model_dir}")


def read_cameras_txt(path: Path):
    cams = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(v) for v in parts[4:]]
            cams[cam_id] = {
                "camera_id": cam_id,
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    return cams


def read_cameras_bin(path: Path):
    cams = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]

            model_num_params = {
                0: 3,
                1: 4,
                2: 4,
                3: 5,
                4: 8,
                5: 8,
                6: 12,
                7: 5,
                8: 4,
                9: 5,
                10: 12,
            }
            nparams = model_num_params[model_id]
            params = list(struct.unpack("<" + "d" * nparams, f.read(8 * nparams)))

            model_names = {
                0: "SIMPLE_PINHOLE",
                1: "PINHOLE",
                2: "SIMPLE_RADIAL",
                3: "RADIAL",
                4: "OPENCV",
                5: "OPENCV_FISHEYE",
                6: "FULL_OPENCV",
                7: "FOV",
                8: "SIMPLE_RADIAL_FISHEYE",
                9: "RADIAL_FISHEYE",
                10: "THIN_PRISM_FISHEYE",
            }

            cams[camera_id] = {
                "camera_id": camera_id,
                "model": model_names.get(model_id, f"MODEL_{model_id}"),
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    return cams


def read_images_txt(path: Path):
    images = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue

        image_id = int(parts[0])
        qvec = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
        tvec = [float(parts[5]), float(parts[6]), float(parts[7])]
        camera_id = int(parts[8])
        name = parts[9]

        images[image_id] = {
            "image_id": image_id,
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": camera_id,
            "name": name,
        }
        i += 2

    return images


def read_images_bin(path: Path):
    images = {}
    with open(path, "rb") as f:
        num_reg_images = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_reg_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<dddd", f.read(32))
            tvec = struct.unpack("<ddd", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]

            name_bytes = bytearray()
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                if ch == b"":
                    raise ValueError("Fin de fichier inattendue dans images.bin")
                name_bytes.extend(ch)
            name = name_bytes.decode("utf-8")

            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2D * 24)

            images[image_id] = {
                "image_id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            }
    return images


def read_points3d_txt(path: Path):
    pts = []
    cols = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            r = int(parts[4])
            g = int(parts[5])
            b = int(parts[6])
            pts.append([x, y, z])
            cols.append([r, g, b])
    return np.asarray(pts, dtype=np.float64), np.asarray(cols, dtype=np.uint8)


def read_points3d_bin(path: Path):
    pts = []
    cols = []
    with open(path, "rb") as f:
        num_points_data = f.read(8)
        if len(num_points_data) != 8:
            raise ValueError("Fichier points3D.bin invalide ou vide")
        num_points = struct.unpack("<Q", num_points_data)[0]

        for _ in range(num_points):
            point3d_id_data = f.read(8)
            xyz_data = f.read(24)
            rgb_data = f.read(3)
            error_data = f.read(8)
            track_len_data = f.read(8)

            if (
                len(point3d_id_data) != 8
                or len(xyz_data) != 24
                or len(rgb_data) != 3
                or len(error_data) != 8
                or len(track_len_data) != 8
            ):
                raise ValueError("Fichier points3D.bin tronqué")

            _point3d_id = struct.unpack("<Q", point3d_id_data)[0]
            x, y, z = struct.unpack("<ddd", xyz_data)
            r, g, b = struct.unpack("<BBB", rgb_data)
            _error = struct.unpack("<d", error_data)[0]
            track_length = struct.unpack("<Q", track_len_data)[0]

            f.read(track_length * 8)

            pts.append([x, y, z])
            cols.append([r, g, b])

    return np.asarray(pts, dtype=np.float64), np.asarray(cols, dtype=np.uint8)


def get_intrinsics_from_camera(cam):
    model = cam["model"]
    p = cam["params"]

    if model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
    elif model == "SIMPLE_PINHOLE":
        f, cx, cy = p[:3]
        fx = fy = f
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, _k1 = p[:4]
        fx = fy = f
    elif model == "RADIAL":
        f, cx, cy, _k1, _k2 = p[:5]
        fx = fy = f
    elif model == "OPENCV":
        fx, fy, cx, cy = p[:4]
    elif model == "FULL_OPENCV":
        fx, fy, cx, cy = p[:4]
    else:
        raise NotImplementedError(f"Modèle caméra non supporté pour transforms.json: {model}")

    return {
        "w": int(cam["width"]),
        "h": int(cam["height"]),
        "fl_x": float(fx),
        "fl_y": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "camera_model": "PINHOLE",
    }


def build_T_wc_from_colmap_image(colmap_image):
    qvec = np.asarray(colmap_image["qvec"], dtype=np.float64)
    tvec = np.asarray(colmap_image["tvec"], dtype=np.float64)

    R_cw = qvec_to_rotmat(qvec)
    R_wc = R_cw.T
    C = -R_wc @ tvec

    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = C
    return T_wc


def apply_3x4_transform(points_xyz: np.ndarray, applied_transform):
    if applied_transform is None:
        return np.asarray(points_xyz, dtype=np.float32)

    T = np.asarray(applied_transform, dtype=np.float32)
    if T.shape != (3, 4):
        raise ValueError(f"applied_transform doit être de forme (3,4), reçu {T.shape}")

    R = T[:, :3]
    t = T[:, 3]

    pts = np.asarray(points_xyz, dtype=np.float32)
    return (pts @ R.T) + t[None, :]


def write_sparse_pc_ply_with_nerfstudio_logic(
    filename: str,
    recon_dir: Path,
    output_dir: Path,
    applied_transform=None,
) -> None:
    """
    Écrit sparse_pc.ply en reproduisant à la main la logique de
    nerfstudio.process_data.colmap_utils.create_ply_from_colmap,
    sans dépendre de torch ni de nerfstudio.
    """
    pts_xyz, pts_rgb = load_colmap_points3D(recon_dir)

    points3D = apply_3x4_transform(pts_xyz, applied_transform)
    points3D_rgb = np.asarray(pts_rgb, dtype=np.uint8)

    with open(output_dir / filename, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points3D)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uint8 red\n")
        f.write("property uint8 green\n")
        f.write("property uint8 blue\n")
        f.write("end_header\n")

        for coord, color in zip(points3D, points3D_rgb):
            x, y, z = coord
            r, g, b = color
            f.write(f"{float(x):8f} {float(y):8f} {float(z):8f} {int(r)} {int(g)} {int(b)}\n")


def write_transforms_json(path: Path, cameras: dict, images: dict, ply_file_name="sparse_pc.ply"):
    if not images:
        raise ValueError("Aucune image COLMAP pour écrire transforms.json")

    first_image = next(iter(images.values()))
    first_cam = cameras[first_image["camera_id"]]
    intr = get_intrinsics_from_camera(first_cam)

    data = {
        "w": intr["w"],
        "h": intr["h"],
        "fl_x": intr["fl_x"],
        "fl_y": intr["fl_y"],
        "cx": intr["cx"],
        "cy": intr["cy"],
        "camera_model": intr["camera_model"],
        "ply_file_path": ply_file_name,
        "applied_transform": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        "applied_scale": 1.0,
        "frames": [],
    }

    for image_id, im in sorted(images.items()):
        T_wc = build_T_wc_from_colmap_image(im)
        data["frames"].append({
            "file_path": f"./images/{im['name']}",
            "transform_matrix": T_wc.tolist(),
            "colmap_im_id": int(image_id),
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    ap = argparse.ArgumentParser(
        description="Prépare un modèle COLMAP à partir d'un dossier d'images et écrit transforms.json + sparse_pc.ply."
    )
    ap.add_argument("--images", required=True, help="Dossier d'images en entrée")
    ap.add_argument("--out", required=True, help="Dossier de sortie")
    ap.add_argument("--colmap", default=None, help="Chemin vers le binaire colmap")
    ap.add_argument("--camera-model", default="OPENCV", help="Modèle caméra COLMAP")
    ap.add_argument("--single-camera", action="store_true", help="Utiliser une seule caméra pour toutes les images")
    ap.add_argument("--camera-params", default=None, help="Paramètres caméra COLMAP si connus")
    ap.add_argument("--matcher", default="exhaustive", choices=["exhaustive", "sequential", "spatial"], help="Type de matching")
    ap.add_argument("--gpu", action="store_true", help="Utiliser le GPU pour COLMAP si disponible")
    ap.add_argument("--skip-undistort", action="store_true", help="Ne pas lancer image_undistorter")
    args = ap.parse_args()

    images_dir = Path(args.images)
    out_dir = Path(args.out)

    if not images_dir.exists():
        raise FileNotFoundError(f"Dossier images introuvable: {images_dir}")

    colmap_bin = find_colmap_binary(args.colmap)

    db_path = out_dir / "database.db"
    sparse_dir = out_dir / "colmap" / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    undistort_dir = out_dir / "undistorted"
    images_out_dir = out_dir / "images"

    if images_out_dir.exists():
        warn(f"Le dossier de sortie images existe déjà: {images_out_dir}")
    else:
        shutil.copytree(images_dir, images_out_dir)

    use_gpu = "1" if args.gpu else "0"

    info("=== Étape 1: feature_extractor ===")
    cmd = [
        colmap_bin, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_out_dir),
        "--ImageReader.camera_model", str(args.camera_model),
        "--SiftExtraction.use_gpu", use_gpu,
    ]
    if args.single_camera:
        cmd += ["--ImageReader.single_camera", "1"]
    if args.camera_params:
        cmd += ["--ImageReader.camera_params", str(args.camera_params)]
    run_cmd(cmd)

    info("=== Étape 2: matching ===")
    if args.matcher == "exhaustive":
        cmd = [
            colmap_bin, "exhaustive_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", use_gpu,
        ]
    elif args.matcher == "sequential":
        cmd = [
            colmap_bin, "sequential_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", use_gpu,
        ]
    else:
        cmd = [
            colmap_bin, "spatial_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", use_gpu,
        ]
    run_cmd(cmd)

    info("=== Étape 3: mapper ===")
    cmd = [
        colmap_bin, "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_out_dir),
        "--output_path", str(sparse_dir),
    ]
    run_cmd(cmd)

    model_dirs = sorted([p for p in sparse_dir.iterdir() if p.is_dir()])
    if not model_dirs:
        raise RuntimeError("Aucun modèle sparse produit par COLMAP.")
    sparse_model_dir = model_dirs[0]
    info(f"Modèle sparse retenu: {sparse_model_dir}")

    info("=== Étape 4: model_converter (TXT) ===")
    txt_model_dir = out_dir / "colmap" / "sparse_txt"
    txt_model_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        colmap_bin, "model_converter",
        "--input_path", str(sparse_model_dir),
        "--output_path", str(txt_model_dir),
        "--output_type", "TXT",
    ]
    run_cmd(cmd)

    if not args.skip_undistort:
        info("=== Étape 5: image_undistorter ===")

        if undistort_dir.exists():
            warn(f"Suppression de l'ancien dossier undistorted: {undistort_dir}")
            shutil.rmtree(undistort_dir)

        undistort_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            colmap_bin, "image_undistorter",
            "--image_path", str(images_out_dir),
            "--input_path", str(sparse_model_dir),
            "--output_path", str(undistort_dir),
            "--output_type", "COLMAP",
        ]
        run_cmd(cmd)

    info("=== Étape 6: lecture du modèle COLMAP ===")
    cameras = load_colmap_cameras(sparse_model_dir)
    images = load_colmap_images(sparse_model_dir)
    pts_xyz, _pts_rgb = load_colmap_points3D(sparse_model_dir)

    info(f"Cameras  : {len(cameras)}")
    info(f"Images   : {len(images)}")
    info(f"Points3D : {len(pts_xyz)}")

    info("=== Étape 7: écriture sparse_pc.ply ===")
    ply_path = out_dir / "sparse_pc.ply"
    write_sparse_pc_ply_with_nerfstudio_logic(
        filename="sparse_pc.ply",
        recon_dir=sparse_model_dir,
        output_dir=out_dir,
        applied_transform=None,
    )

    info("=== Étape 8: écriture transforms.json ===")
    transforms_path = out_dir / "transforms.json"
    write_transforms_json(transforms_path, cameras, images, ply_file_name="sparse_pc.ply")

    info("=== Terminé ===")
    print(f"Images output        : {images_out_dir}")
    print(f"Database             : {db_path}")
    print(f"Sparse model         : {sparse_model_dir}")
    print(f"Sparse TXT           : {txt_model_dir}")
    print(f"PLY                  : {ply_path}")
    print(f"Transforms           : {transforms_path}")
    if not args.skip_undistort:
        print(f"Undistorted output   : {undistort_dir}")


if __name__ == "__main__":
    main()

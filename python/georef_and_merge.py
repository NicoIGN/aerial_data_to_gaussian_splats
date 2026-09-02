#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import struct
import shutil
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET

try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False


# -----------------------------
# Utilitaires COLMAP
# -----------------------------
def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    n = np.linalg.norm(qvec)
    if n == 0:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qvec / n

    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy]
    ], dtype=np.float64)


def _read_cstring(f):
    chars = []
    while True:
        c = f.read(1)
        if c == b"":
            raise EOFError("EOF pendant lecture string C")
        if c == b"\x00":
            break
        chars.append(c)
    return b"".join(chars).decode("utf-8", errors="replace")


def read_colmap_image_centers_from_bin(images_bin_path):
    centers = {}
    with open(images_bin_path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            _image_id = struct.unpack("<i", f.read(4))[0]
            qvec = np.array(struct.unpack("<4d", f.read(32)), dtype=np.float64)
            tvec = np.array(struct.unpack("<3d", f.read(24)), dtype=np.float64)
            _camera_id = struct.unpack("<i", f.read(4))[0]
            name = _read_cstring(f)

            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.seek(num_points2d * 24, 1)

            R = qvec_to_rotmat(qvec)
            C = -R.T @ tvec
            centers[Path(name).stem.lower()] = C
    return centers


def read_colmap_image_centers_from_txt(images_txt_path):
    centers = {}
    with open(images_txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 10:
            continue

        try:
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            name = " ".join(parts[9:])
        except ValueError:
            continue

        R = qvec_to_rotmat(np.array([qw, qx, qy, qz], dtype=np.float64))
        t = np.array([tx, ty, tz], dtype=np.float64)
        C = -R.T @ t
        centers[Path(name).stem.lower()] = C

        if i < len(lines):
            i += 1
    return centers


def resolve_sparse_dir(colmap_output: Path) -> Path:
    candidates = [
        colmap_output / "colmap" / "sparse" / "0",  # nouveau layout
        colmap_output / "sparse" / "0",             # ancien layout
    ]
    for d in candidates:
        if (d / "images.bin").exists() or (d / "images.txt").exists():
            return d
    raise FileNotFoundError(
        "Aucun fichier COLMAP images trouvé. Attendu: "
        + " ou ".join(str(d / "images.bin") for d in candidates)
    )


def read_colmap_image_centers(colmap_output_dir):
    colmap_output_dir = Path(colmap_output_dir)
    sparse_dir = resolve_sparse_dir(colmap_output_dir)

    cand_bin = sparse_dir / "images.bin"
    if cand_bin.exists():
        return read_colmap_image_centers_from_bin(cand_bin)

    cand_txt_sparse = sparse_dir / "images.txt"
    if cand_txt_sparse.exists():
        return read_colmap_image_centers_from_txt(cand_txt_sparse)

    raise FileNotFoundError(f"Aucun images.bin/images.txt dans {sparse_dir}")


# -----------------------------
# Utilitaires .CON
# -----------------------------
def read_con(path):
    root = ET.parse(path).getroot()

    def val(xpath, default=None):
        node = root.find(xpath)
        if node is None or node.text is None:
            return default
        return float(node.text)

    # Même convention que ton pipeline principal (euclidien x/y + altitude)
    return np.array([
        val(".//extrinseque/systeme/euclidien/x", 0.0),
        val(".//extrinseque/systeme/euclidien/y", 0.0),
        val(".//extrinseque/sommet/altitude", 0.0),
    ], dtype=np.float64)


def read_con_centers(image_dir):
    con_files = sorted(image_dir.glob("*.CON")) + sorted(image_dir.glob("*.con"))
    out = {}
    for con_path in con_files:
        out[con_path.stem.lower()] = read_con(con_path)
    return out


def build_correspondences(colmap_centers_dict, con_centers_dict):
    common = sorted(set(colmap_centers_dict.keys()) & set(con_centers_dict.keys()))
    if len(common) == 0:
        return [], np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    A = np.array([colmap_centers_dict[k] for k in common], dtype=np.float64)
    B = np.array([con_centers_dict[k] for k in common], dtype=np.float64)
    return common, A, B


# -----------------------------
# Estimation rigide
# -----------------------------
def estimate_rigid_transform(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape != B.shape or A.shape[1] != 3:
        raise ValueError("A et B doivent avoir forme (N,3)")
    if len(A) < 3:
        raise ValueError("Au moins 3 correspondances requises")

    ca = A.mean(axis=0)
    cb = B.mean(axis=0)
    A0 = A - ca
    B0 = B - cb

    H = A0.T @ B0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = cb - R @ ca
    A_aligned = (R @ A.T).T + t
    residuals = np.linalg.norm(A_aligned - B, axis=1)
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    return R, t, rms, residuals, A_aligned


# -----------------------------
# I/O PLY
# -----------------------------
def read_ply_xyzrgb(path):
    xyz, rgb = [], []
    try:
        with open(path, "r", encoding="utf-8") as f:
            header_end = False
            n_vertices = 0
            is_ascii = True
            for line in f:
                if line.startswith("format") and "binary" in line:
                    is_ascii = False
                    break
                if line.startswith("element vertex"):
                    n_vertices = int(line.split()[-1])
                if line.startswith("end_header"):
                    header_end = True
                    break
            if not header_end:
                raise ValueError("header invalide")
            if not is_ascii:
                raise ValueError("binary")

            for _ in range(n_vertices):
                line = f.readline()
                if not line:
                    break
                p = line.split()
                if len(p) >= 3:
                    xyz.append([float(p[0]), float(p[1]), float(p[2])])
                    if len(p) >= 6:
                        rgb.append([int(p[3]), int(p[4]), int(p[5])])

        xyz = np.array(xyz, dtype=np.float64)
        rgb = np.array(rgb, dtype=np.uint8) if len(rgb) == len(xyz) and len(rgb) > 0 else None
        return xyz, rgb
    except (UnicodeDecodeError, ValueError):
        return read_ply_binary(path)


def read_ply_binary(path):
    xyz, rgb = [], []
    with open(path, "rb") as f:
        header_end = False
        n_vertices = 0
        is_little_endian = True

        while not header_end:
            raw = f.readline()
            if not raw:
                raise ValueError(f"Header PLY incomplet: {path}")
            line = raw.decode("utf-8", errors="ignore").strip()
            if line.startswith("format"):
                is_little_endian = "little_endian" in line
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            if line == "end_header":
                header_end = True

        endian = "<" if is_little_endian else ">"
        fmt = endian + "fffBBB"
        size = struct.calcsize(fmt)

        for _ in range(n_vertices):
            data = f.read(size)
            if len(data) < size:
                break
            x, y, z, r, g, b = struct.unpack(fmt, data)
            xyz.append([x, y, z])
            rgb.append([r, g, b])

    return np.array(xyz, dtype=np.float64), np.array(rgb, dtype=np.uint8) if len(rgb) else None


def write_ply_xyzrgb(path, xyz, rgb=None):
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    if rgb is None:
        rgb = np.full((n, 3), 200, dtype=np.uint8)
    else:
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb):
            f.write(f"{p[0]:.12f} {p[1]:.12f} {p[2]:.12f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


# -----------------------------
# I/O LAZ
# -----------------------------
def read_laz_points(laz_path, stride=1, xmin=None, xmax=None, ymin=None, ymax=None):
    las = laspy.read(str(laz_path))
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)

    mask = np.ones(len(xyz), dtype=bool)
    if xmin is not None:
        mask &= xyz[:, 0] >= float(xmin)
    if xmax is not None:
        mask &= xyz[:, 0] <= float(xmax)
    if ymin is not None:
        mask &= xyz[:, 1] >= float(ymin)
    if ymax is not None:
        mask &= xyz[:, 1] <= float(ymax)

    xyz = xyz[mask]

    rgb = None
    if all(hasattr(las, a) for a in ("red", "green", "blue")):
        rgb = np.vstack((las.red, las.green, las.blue)).T[mask]
        rgb = np.clip(rgb / 256.0, 0, 255).astype(np.uint8) if rgb.size > 0 else None

    stride = max(1, int(stride))
    return xyz[::stride], (rgb[::stride] if rgb is not None else None)


def finalize_sparse_pc_ply(out_dir, verbose=False):
    out_dir = Path(out_dir)
    sparse_pc = out_dir / "sparse_pc.ply"
    fused = out_dir / "model_fused.ply"
    model = out_dir / "model.ply"

    if fused.exists():
        shutil.copy2(fused, sparse_pc)
        if verbose:
            print(f"[OK] sparse_pc.ply <- {fused.name}")
        return True
    if model.exists():
        shutil.copy2(model, sparse_pc)
        if verbose:
            print(f"[OK] sparse_pc.ply <- {model.name}")
        return True
    return False


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Géoréférence PLY COLMAP + fusion LiDAR optionnelle")

    parser.add_argument("--colmap-output", required=True, help="Dossier output pipeline")
    parser.add_argument("--laz", default=None, help="Fichier .LAZ lidar (optionnel)")
    parser.add_argument("--image-dir", required=True, help="Dossier contenant .CON (utiliser output/images)")
    parser.add_argument("--out", required=True, help="Dossier de sortie")
    parser.add_argument("--subsample", type=int, default=10, help="Sous-échantillonnage LAZ")
    parser.add_argument("--xmin", type=float, default=None)
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    colmap_output = Path(args.colmap_output)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out)
    laz_path = Path(args.laz).resolve() if args.laz else None

    if not colmap_output.exists():
        print(f"[ERROR] Dossier COLMAP inexistant: {colmap_output}")
        sys.exit(1)
    if not image_dir.exists():
        print(f"[ERROR] Dossier images/.CON inexistant: {image_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    sparse_dir = resolve_sparse_dir(colmap_output)
    ply_relative = colmap_output / "model.ply"

    if args.verbose:
        print(f"[INFO] Dossier COLMAP: {colmap_output}")
        print(f"[INFO] Sparse résolu: {sparse_dir}")
        print(f"[INFO] Fichier LAZ: {laz_path if laz_path else 'AUCUN'}")
        print(f"[INFO] Dossier images/.CON: {image_dir}")
        print(f"[INFO] Sortie: {out_dir}")

    if not ply_relative.exists():
        print(f"[ERROR] PLY COLMAP inexistant: {ply_relative}")
        sys.exit(1)

    # 1) Lire PLY COLMAP
    if args.verbose:
        print("\n[1/5] Lecture du PLY COLMAP relatif...")
    pts_colmap, rgb_colmap = read_ply_xyzrgb(ply_relative)
    if args.verbose:
        print(f"  {len(pts_colmap)} points COLMAP chargés")

    # 2) Lire centres COLMAP + .CON
    if args.verbose:
        print("\n[2/5] Lecture centres caméras COLMAP + poses .CON...")
    try:
        colmap_centers_dict = read_colmap_image_centers(colmap_output)
    except Exception as e:
        print(f"[ERROR] Lecture des caméras COLMAP impossible: {e}")
        sys.exit(1)

    try:
        con_centers_dict = read_con_centers(image_dir)
    except Exception as e:
        print(f"[ERROR] Lecture .CON impossible: {e}")
        sys.exit(1)

    names, A_colmap, B_geo = build_correspondences(colmap_centers_dict, con_centers_dict)
    if len(names) < 3:
        print("[ERROR] Correspondances insuffisantes entre COLMAP et .CON (>=3 requis)")
        print(f"        COLMAP={len(colmap_centers_dict)}, CON={len(con_centers_dict)}, communs={len(names)}")
        sys.exit(1)

    if args.verbose:
        print(f"  Caméras COLMAP lues: {len(colmap_centers_dict)}")
        print(f"  Poses .CON lues:      {len(con_centers_dict)}")
        print(f"  Correspondances:      {len(names)}")

    # 3) Estimer R,t
    if args.verbose:
        print("\n[3/5] Estimation transformation rigide (R,t) + résidus...")
    R, t, rms, residuals, _ = estimate_rigid_transform(A_colmap, B_geo)

    print("\n[ALIGN] Transformation COLMAP -> GEO")
    print("R =")
    print(R)
    print(f"t = {t}")
    print(f"det(R) = {np.linalg.det(R):.6f}")
    print(f"RMS résidu = {rms:.4f} m")
    print(f"Min/Max résidu = {residuals.min():.4f} / {residuals.max():.4f} m")

    print("\n[RESIDUS] Caméra -> Sommet .CON (m)")
    for i, (name, r) in enumerate(zip(names, residuals)):
        print(f"  #{i:04d} {name}: {r:.4f}")

    pts_colmap_geo = (R @ pts_colmap.T).T + t

    # 4/5) LiDAR optionnel
    lidar_used = False
    pts_merged = pts_colmap_geo
    rgb_merged = rgb_colmap if rgb_colmap is not None else np.full((len(pts_colmap_geo), 3), [255, 80, 80], dtype=np.uint8)

    if laz_path is not None and laz_path.exists():
        if not LASPY_AVAILABLE:
            print("[ERROR] laspy non installé alors qu'un LAZ est fourni. Installe: pip install laspy[lazrs]")
            sys.exit(1)

        if args.verbose:
            print("\n[4/5] Lecture du LAZ lidar...")
        pts_lidar, rgb_lidar = read_laz_points(
            laz_path,
            stride=args.subsample,
            xmin=args.xmin, xmax=args.xmax,
            ymin=args.ymin, ymax=args.ymax
        )

        if len(pts_lidar) == 0:
            print("[WARNING] Aucun point lidar après filtrage, fusion ignorée.")
        else:
            if args.verbose:
                print(f"  {len(pts_lidar)} points lidar chargés")

            if rgb_lidar is None:
                rgb_lidar = np.full((len(pts_lidar), 3), [80, 180, 255], dtype=np.uint8)

            pts_merged = np.vstack([pts_colmap_geo, pts_lidar])
            rgb_col = rgb_colmap if rgb_colmap is not None else np.full((len(pts_colmap_geo), 3), [255, 80, 80], dtype=np.uint8)
            rgb_merged = np.vstack([rgb_col, rgb_lidar])
            lidar_used = True
    else:
        if args.verbose:
            print("\n[4/5] Pas de LiDAR fourni -> on garde COLMAP comme référence")

    # 5) Sorties
    if args.verbose:
        print("\n[5/5] Ecriture des sorties...")

    # model.ply géoréférencé (toujours)
    model_geo = out_dir / "model.ply"
    write_ply_xyzrgb(model_geo, pts_colmap_geo, rgb_colmap)

    # fused optionnel
    if lidar_used:
        output_fused = out_dir / "model_fused.ply"
        write_ply_xyzrgb(output_fused, pts_merged, rgb_merged)
        print(f"[OK] PLY fusionné: {output_fused}")

    residuals_csv = out_dir / "alignment_residuals.csv"
    with open(residuals_csv, "w", encoding="utf-8") as f:
        f.write("name,residual_m\n")
        for name, r in zip(names, residuals):
            f.write(f"{name},{r:.6f}\n")

    rt_txt = out_dir / "transform_colmap_to_geo.txt"
    with open(rt_txt, "w", encoding="utf-8") as f:
        f.write("# R (3x3)\n")
        for row in R:
            f.write(" ".join(f"{v:.12f}" for v in row) + "\n")
        f.write("# t (3)\n")
        f.write(" ".join(f"{v:.12f}" for v in t) + "\n")
        f.write(f"# rms_residual_m {rms:.6f}\n")

    if not finalize_sparse_pc_ply(out_dir, verbose=True):
        print("[ERROR] Impossible de produire sparse_pc.ply")
        sys.exit(1)

    print(f"[OK] Résidus: {residuals_csv}")
    print(f"[OK] Transformation: {rt_txt}")
    print(f"[DONE] PLY de référence: {out_dir / 'sparse_pc.ply'}")


if __name__ == "__main__":
    main()

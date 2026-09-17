#!/usr/bin/env python3

import argparse
import json
import math
import shutil
import sqlite3
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import laspy
except ImportError:  # optional
    laspy = None


# ============================================================================
# LOG
# ============================================================================

def log(msg):
    print(msg, flush=True)


def v_log(verbose, msg):
    if verbose:
        print(msg, flush=True)


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def ensure_finite_array(name, arr, verbose=True):
    arr = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(arr).all()
    if not finite:
        bad = np.where(~np.isfinite(arr))
        log(f"[WARN] {name}: contient des NaN/Inf aux indices {bad}")
    return finite


def filter_finite_points(xyz: np.ndarray, rgb: np.ndarray | None = None):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz doit être de forme (N, 3), reçu {xyz.shape}")

    mask = np.isfinite(xyz).all(axis=1)
    xyz2 = xyz[mask]

    rgb2 = None
    if rgb is not None:
        rgb = np.asarray(rgb)
        if len(rgb) != len(mask):
            raise ValueError(f"rgb longueur incompatible: {len(rgb)} vs {len(mask)}")
        rgb2 = rgb[mask]

    return xyz2, rgb2, mask


def summarize_points3d(xyz: np.ndarray, label: str):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"{label}: xyz doit être de forme (N, 3), reçu {xyz.shape}")

    finite_mask = np.isfinite(xyz).all(axis=1)
    valid = xyz[finite_mask]
    invalid = len(xyz) - len(valid)

    log(f"[POINTS3D][{label}] total={len(xyz)} valid={len(valid)} invalid={invalid}")

    if len(valid) == 0:
        log(f"[POINTS3D][{label}] Aucun point valide")
        return {
            "n_total": len(xyz),
            "n_valid": 0,
            "n_invalid": invalid,
            "bbox_min": None,
            "bbox_max": None,
            "bbox_diag": None,
            "mean": None,
        }

    mn = valid.min(axis=0)
    mx = valid.max(axis=0)
    diag = float(np.linalg.norm(mx - mn))
    mean = valid.mean(axis=0)

    log(f"[POINTS3D][{label}] bbox min={mn}")
    log(f"[POINTS3D][{label}] bbox max={mx}")
    log(f"[POINTS3D][{label}] bbox diag={diag:.6f}")
    log(f"[POINTS3D][{label}] mean={mean}")

    return {
        "n_total": len(xyz),
        "n_valid": len(valid),
        "n_invalid": invalid,
        "bbox_min": mn,
        "bbox_max": mx,
        "bbox_diag": diag,
        "mean": mean,
    }


def read_points3d_txt(points3d_txt):
    points3d_txt = Path(points3d_txt)
    pts = []
    with open(points3d_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) < 8:
                continue
            try:
                pts.append([float(toks[1]), float(toks[2]), float(toks[3])])
            except Exception:
                continue
    return np.array(pts, dtype=np.float64) if pts else np.zeros((0, 3), dtype=np.float64)


def read_ply_points(path: Path):
    """
    Lit un PLY ASCII ou binaire et renvoie les positions xyz.
    """
    path = Path(path)
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float64)

    with open(path, "rb") as f:
        fmt = None
        n_verts = None
        vertex_props = []
        in_vertex_block = False

        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"{path}: header PLY tronqué")
            s = line.decode("ascii", errors="ignore").strip()

            if s.startswith("format "):
                fmt = s.split()[1]
            elif s.startswith("element vertex "):
                n_verts = int(s.split()[-1])
                in_vertex_block = True
            elif s.startswith("element ") and not s.startswith("element vertex "):
                in_vertex_block = False
            elif s.startswith("property ") and in_vertex_block:
                toks = s.split()
                if len(toks) >= 3:
                    vertex_props.append((toks[1], toks[2]))
            elif s == "end_header":
                break

        if fmt is None or n_verts is None:
            raise RuntimeError(f"{path}: format ou nombre de vertices manquant")

        ply_to_struct = {
            "char": "b", "int8": "b",
            "uchar": "B", "uint8": "B",
            "short": "h", "int16": "h",
            "ushort": "H", "uint16": "H",
            "int": "i", "int32": "i",
            "uint": "I", "uint32": "I",
            "float": "f", "float32": "f",
            "double": "d", "float64": "d",
        }

        if fmt == "ascii":
            pts = []
            for i in range(n_verts):
                line = f.readline()
                if not line:
                    break
                toks = line.decode("ascii", errors="ignore").split()
                if len(toks) < 3:
                    continue
                try:
                    x, y, z = float(toks[0]), float(toks[1]), float(toks[2])
                    pts.append([x, y, z])
                except Exception:
                    continue
            return np.array(pts, dtype=np.float64) if pts else np.zeros((0, 3), dtype=np.float64)

        endian = "<" if fmt == "binary_little_endian" else ">"
        fmt_chars = []
        x_idx = y_idx = z_idx = None

        for idx, (ptype, pname) in enumerate(vertex_props):
            if ptype not in ply_to_struct:
                raise RuntimeError(f"{path}: type PLY non supporté: {ptype}")
            fmt_chars.append(ply_to_struct[ptype])
            if pname == "x":
                x_idx = idx
            elif pname == "y":
                y_idx = idx
            elif pname == "z":
                z_idx = idx

        if x_idx is None or y_idx is None or z_idx is None:
            raise RuntimeError(f"{path}: propriétés x/y/z manquantes dans le header")

        vertex_struct = struct.Struct(endian + "".join(fmt_chars))
        vertex_size = vertex_struct.size

        pts = []
        for i in range(n_verts):
            raw = f.read(vertex_size)
            if len(raw) < vertex_size:
                break
            try:
                vals = vertex_struct.unpack(raw)
                x, y, z = vals[x_idx], vals[y_idx], vals[z_idx]
                pts.append([x, y, z])
            except Exception:
                continue

        return np.array(pts, dtype=np.float64) if pts else np.zeros((0, 3), dtype=np.float64)


def read_text_images_centers(images_txt):
    lines = Path(images_txt).read_text(encoding="utf-8").splitlines()
    out = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        toks = line.split()
        if len(toks) < 10:
            i += 1
            continue

        q = np.array([float(toks[1]), float(toks[2]), float(toks[3]), float(toks[4])], dtype=np.float64)
        t = np.array([float(toks[5]), float(toks[6]), float(toks[7])], dtype=np.float64)
        name = " ".join(toks[9:])

        R = quaternion_to_rotmat(q)
        C = translation_to_center(R, t)
        out[Path(name).name] = C
        i += 2
    return out


def rotmat_to_quaternion(R):
    trace = np.trace(R)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Quaternion nul")
    q /= norm
    return q


def quaternion_to_rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def camera_center_to_translation(R, C):
    return -R @ C


def translation_to_center(R, t):
    return -R.T @ t


# ============================================================================
# PARTIE 1: LECTURE .CON
# ============================================================================

def read_con(path, jpg_filename=None):
    root = ET.parse(path).getroot()

    def val(xpath, default=None):
        node = root.find(xpath)
        if node is None or node.text is None:
            return default
        return float(node.text)

    image_name = jpg_filename if jpg_filename is not None else (root.findtext(".//image_name") or path.stem)

    A = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float64)

    center_raw = np.array([
        val(".//extrinseque/systeme/euclidien/x"),
        val(".//extrinseque/systeme/euclidien/y"),
        val(".//extrinseque/sommet/altitude"),
    ], dtype=np.float64)

    R_raw = np.array([
        [val(".//rotation/mat3d/l1/pt3d/x"), val(".//rotation/mat3d/l1/pt3d/y"), val(".//rotation/mat3d/l1/pt3d/z")],
        [val(".//rotation/mat3d/l2/pt3d/x"), val(".//rotation/mat3d/l2/pt3d/y"), val(".//rotation/mat3d/l2/pt3d/z")],
        [val(".//rotation/mat3d/l3/pt3d/x"), val(".//rotation/mat3d/l3/pt3d/y"), val(".//rotation/mat3d/l3/pt3d/z")],
    ], dtype=np.float64)

    center = A @ center_raw
    R = R_raw @ A.T

    return {
        "image_name": image_name,
        "center": center,
        "R": R,
        "width": int(val(".//intrinseque/sensor/image_size/width")),
        "height": int(val(".//intrinseque/sensor/image_size/height")),
        "focal": val(".//intrinseque/sensor/ppa/focale"),
        "cx": val(".//intrinseque/sensor/ppa/c"),
        "cy": val(".//intrinseque/sensor/ppa/l"),
        "r1": val(".//intrinseque/sensor/distortion/r1", 0.0),
        "r3": val(".//intrinseque/sensor/distortion/r3", 0.0),
        "r5": val(".//intrinseque/sensor/distortion/r5", 0.0),
        "r7": val(".//intrinseque/sensor/distortion/r7", 0.0),
    }


def resolve_sparse_dir(colmap_output: Path) -> Path:
    candidates = [
        colmap_output / "colmap" / "sparse" / "0",
        colmap_output / "sparse" / "0",
    ]
    for d in candidates:
        if (d / "images.bin").exists() or (d / "images.txt").exists():
            return d
    raise RuntimeError("Aucun modèle COLMAP trouvé. Attendu: " + " ou ".join(str(d / "images.bin") for d in candidates))


def prepare_output_images_dir(src_images_dir, output_dir, verbose=False):
    src_images_dir = Path(src_images_dir).resolve()
    out_images = Path(output_dir).resolve() / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    img_files = []
    for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
        img_files.extend(src_images_dir.glob(ext))
    img_files = sorted(img_files)

    if not img_files:
        raise RuntimeError(f"Aucune image trouvée dans {src_images_dir}")

    copied_img = 0
    copied_con = 0

    for img in img_files:
        shutil.copy2(img, out_images / img.name)
        copied_img += 1

        con_upper = src_images_dir / f"{img.stem}.CON"
        con_lower = src_images_dir / f"{img.stem}.con"

        if con_upper.exists():
            shutil.copy2(con_upper, out_images / con_upper.name)
            copied_con += 1
        elif con_lower.exists():
            shutil.copy2(con_lower, out_images / con_lower.name)
            copied_con += 1

    v_log(verbose, f"[IMAGES] Copiées: {copied_img} images, {copied_con} .CON -> {out_images}")

    if copied_con == 0:
        log(
            f"[WARN] Aucun fichier .CON/.con trouvé à côté des images dans "
            f"{src_images_dir}. Vérifie que les .CON sont bien présents "
            f"dans ce dossier (pas seulement les .jpg)."
        )

    return out_images


def _find_con_for_image(output_images_dir: Path, stem: str):
    con_u = output_images_dir / f"{stem}.CON"
    if con_u.exists():
        return con_u
    con_l = output_images_dir / f"{stem}.con"
    if con_l.exists():
        return con_l
    return None


# ============================================================================
# PARTIE 2: KEYPOINTS / FILTERS
# ============================================================================

def extract_keypoints(image_path, detector_type="SIFT", max_keypoints=None):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Impossible de charger l'image: {image_path}")

    if detector_type == "SIFT":
        detector = cv2.SIFT_create()
    elif detector_type == "ORB":
        detector = cv2.ORB_create(nfeatures=max_keypoints or 500)
    elif detector_type == "AKAZE":
        detector = cv2.AKAZE_create()
    else:
        raise ValueError(f"Détecteur inconnu: {detector_type}")

    kp, des = detector.detectAndCompute(img, None)
    if max_keypoints and len(kp) > max_keypoints:
        order = np.argsort([-k.response for k in kp])[:max_keypoints]
        kp = [kp[i] for i in order]
        if des is not None:
            des = des[order]
    return kp, des


def filter_keypoints_by_tolerance(keypoints, con_data, image_path, verbose=False):
    if not keypoints:
        return [], []

    width, height = con_data["width"], con_data["height"]
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Impossible de charger l'image: {image_path}")

    filtered_kp, filtered_idx = [], []
    for idx, kp in enumerate(keypoints):
        x, y = kp.pt
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        if img[int(y), int(x)] < 10:
            continue
        xi, yi = int(x), int(y)
        if 1 < xi < width - 2 and 1 < yi < height - 2:
            gx = abs(int(img[yi, xi + 1]) - int(img[yi, xi - 1]))
            gy = abs(int(img[yi + 1, xi]) - int(img[yi - 1, xi]))
            if np.sqrt(gx * gx + gy * gy) > 20:
                filtered_kp.append(kp)
                filtered_idx.append(idx)
        else:
            if kp.response > 0.01:
                filtered_kp.append(kp)
                filtered_idx.append(idx)

    v_log(verbose, f"    Filtrage: {len(keypoints)} -> {len(filtered_kp)} ({100 * len(filtered_kp) / max(1, len(keypoints)):.1f}%)")
    return filtered_kp, filtered_idx


# ============================================================================
# PARTIE 3: COLMAP DATABASE / MODEL INIT
# ============================================================================

def colmap_model_id(camera_model):
    return {
        "SIMPLE_PINHOLE": 0,
        "PINHOLE": 1,
        "OPENCV": 4,
        "FULL_OPENCV": 6,
    }[camera_model]


def create_colmap_database(colmap_exe, db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    cmd = [colmap_exe, "database_creator", "--database_path=" + str(db_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"database_creator échoué: {r.stderr}")
    log("[OK] Base créée (vide)")


def add_camera_to_database(db_path, records, camera_model="PINHOLE"):
    rec0 = records[0]

    if camera_model == "SIMPLE_PINHOLE":
        params = np.array([rec0["focal"], rec0["cx"], rec0["cy"]], dtype=np.float64)
    elif camera_model == "PINHOLE":
        params = np.array([rec0["focal"], rec0["focal"], rec0["cx"], rec0["cy"]], dtype=np.float64)
    elif camera_model == "OPENCV":
        params = np.array([rec0["focal"], rec0["focal"], rec0["cx"], rec0["cy"], rec0["r1"], rec0["r3"], 0.0, 0.0], dtype=np.float64)
    elif camera_model == "FULL_OPENCV":
        params = np.array([rec0["focal"], rec0["focal"], rec0["cx"], rec0["cy"], rec0["r1"], rec0["r3"], 0.0, 0.0, rec0["r5"], rec0["r7"], 0.0, 0.0], dtype=np.float64)
    else:
        raise ValueError(camera_model)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "UPDATE cameras SET model=?, width=?, height=?, params=? WHERE camera_id=1",
        (colmap_model_id(camera_model), rec0["width"], rec0["height"], params.tobytes()),
    )
    conn.commit()
    conn.close()
    log(f"[OK] Caméra DB mise à jour ({camera_model})")


def write_init_model_from_con(records, image_name_to_id, out_model_dir, camera_model):
    out_model_dir = Path(out_model_dir).resolve()
    out_model_dir.mkdir(parents=True, exist_ok=True)

    rec0 = records[0]
    if camera_model == "SIMPLE_PINHOLE":
        params = f"{rec0['focal']} {rec0['cx']} {rec0['cy']}"
    elif camera_model == "PINHOLE":
        params = f"{rec0['focal']} {rec0['focal']} {rec0['cx']} {rec0['cy']}"
    elif camera_model == "OPENCV":
        params = f"{rec0['focal']} {rec0['focal']} {rec0['cx']} {rec0['cy']} {rec0['r1']} {rec0['r3']} 0 0"
    elif camera_model == "FULL_OPENCV":
        params = f"{rec0['focal']} {rec0['focal']} {rec0['cx']} {rec0['cy']} {rec0['r1']} {rec0['r3']} 0 0 {rec0['r5']} {rec0['r7']} 0 0"
    else:
        raise ValueError(camera_model)

    with open(out_model_dir / "cameras.txt", "w", encoding="utf-8") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 {camera_model} {rec0['width']} {rec0['height']} {params}\n")

    rec_by_name = {r["image_name"]: r for r in records}
    written = 0
    with open(out_model_dir / "images.txt", "w", encoding="utf-8") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for img_name, img_id in sorted(image_name_to_id.items(), key=lambda x: x[1]):
            rec = rec_by_name.get(img_name)
            if rec is None:
                continue
            q = rotmat_to_quaternion(rec["R"])
            t = camera_center_to_translation(rec["R"], rec["center"])
            f.write(
                f"{img_id} {q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {q[3]:.17g} "
                f"{t[0]:.17g} {t[1]:.17g} {t[2]:.17g} 1 {img_name}\n\n"
            )
            written += 1

    with open(out_model_dir / "points3D.txt", "w", encoding="utf-8") as f:
        f.write("# empty init points\n")

    return written


def ensure_init_model_exists_from_db_and_con(db_path, records, sparse_init_dir, camera_model):
    sparse_init_dir = Path(sparse_init_dir).resolve()
    if sparse_init_dir.exists() and (sparse_init_dir / "images.txt").exists():
        return

    log("[INFO] sparse_init absent -> reconstruction depuis .CON")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT image_id, name FROM images ORDER BY image_id")
    rows = cur.fetchall()
    conn.close()

    image_name_to_id = {Path(name).name: image_id for image_id, name in rows}
    n_written = write_init_model_from_con(records, image_name_to_id, sparse_init_dir, camera_model)
    log(f"[OK] Modèle initial reconstruit ({n_written} poses)")


def log_pose_residuals_against_con(con_records, images_txt_after, verbose=False):
    con_map = {r["image_name"]: r["center"] for r in con_records}
    rec_map = read_text_images_centers(images_txt_after)
    common = sorted(set(con_map.keys()) & set(rec_map.keys()))
    if not common:
        log("[RESIDUS] Aucune image commune")
        return

    errs = []
    log("\n[RESIDUS] Ecart centres caméras reconstruits vs .CON")
    for name in common:
        d = rec_map[name] - con_map[name]
        e = float(np.linalg.norm(d))
        errs.append(e)
        if verbose:
            log(f"  {name}: dx={d[0]:+.4f} dy={d[1]:+.4f} dz={d[2]:+.4f} |e|={e:.4f} m")

    errs = np.array(errs, dtype=np.float64)
    log(f"[RESIDUS] N={len(errs)}")
    log(f"[RESIDUS] min={errs.min():.4f} mean={errs.mean():.4f} rms={np.sqrt(np.mean(errs**2)):.4f} max={errs.max():.4f} p95={np.percentile(errs,95):.4f} m")


# ============================================================================
# PARTIE 4: COLMAP RUNNERS / EXPORT
# ============================================================================

def check_colmap_installed(colmap_exe):
    try:
        r = subprocess.run([colmap_exe, "help"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def run_model_converter(colmap_exe, input_path, output_path, output_type):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    cmd = [colmap_exe, "model_converter",
           "--input_path=" + str(input_path),
           "--output_path=" + str(output_path),
           "--output_type=" + output_type]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        err = r.stderr if r.stderr else r.stdout
        raise RuntimeError(f"model_converter échoué ({output_type}):\n{err[:1200]}")

    return output_path


def run_colmap_feature_matching(colmap_exe, database_path, gpu=True):
    log("\n[MATCHING] Appairage des features...")
    cmd = [colmap_exe, "exhaustive_matcher", "--database_path=" + str(database_path)]
    cmd.append("--SiftMatching.use_gpu=true" if gpu else "--SiftMatching.use_gpu=false")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode == 0:
        log("[OK] Appairage terminé")
        return True
    err = r.stderr if r.stderr else r.stdout
    log(f"[ERROR] Appairage échoué:\n{err[:1000]}")
    return False


def run_colmap_point_triangulator(colmap_exe, database_path, image_dir, input_model_dir, output_model_dir):
    log("\n[TRIANGULATION] point_triangulator avec poses initiales .CON...")

    database_path = Path(database_path).resolve()
    image_dir = Path(image_dir).resolve()
    input_model_dir = Path(input_model_dir).resolve()
    output_model_dir = Path(output_model_dir).resolve()
    output_model_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        colmap_exe, "point_triangulator",
        "--database_path=" + str(database_path),
        "--image_path=" + str(image_dir),
        "--input_path=" + str(input_model_dir),
        "--output_path=" + str(output_model_dir),
        "--Mapper.ba_refine_principal_point=0",
        "--Mapper.ba_refine_focal_length=0",
        "--Mapper.ba_refine_extra_params=0",
        "--Mapper.ba_global_use_pba=0",
        "--Mapper.ba_global_max_num_iterations=1",
        "--Mapper.ba_local_max_num_iterations=1",
    ]

    log("[DEBUG] " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.returncode == 0:
        log("[OK] Triangulation terminée")
        return True
    err = r.stderr if r.stderr else r.stdout
    log(f"[ERROR] Triangulation échouée:\n{err[-1500:]}")
    return False


def run_colmap_bundle_adjustment(colmap_exe, model_path):
    log("\n[BUNDLE ADJUSTMENT] Affinage du modèle...")
    model_path = Path(model_path).resolve()
    if not (model_path / "cameras.bin").exists():
        log("[SKIP] Pas de modèle triangulé")
        return True

    cmd = [
        colmap_exe, "bundle_adjuster",
        "--input_path=" + str(model_path),
        "--output_path=" + str(model_path),
        "--BundleAdjustment.refine_principal_point=false",
        "--BundleAdjustment.refine_focal_length=false",
        "--BundleAdjustment.refine_extra_params=false",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode == 0:
        log("[OK] Bundle adjustment terminé")
        return True
    err = r.stderr.strip() if r.stderr else r.stdout.strip()
    log(f"[WARNING] Bundle adjustment non critique échoué:\n{err[:1000]}")
    return True


def export_model_ply(colmap_exe, model_path, output_ply):
    log("\n[EXPORT] Export PLY...")
    model_path = Path(model_path).resolve()
    output_ply = Path(output_ply).resolve()

    if not (model_path / "cameras.bin").exists():
        log("[ERROR] Modèle triangulé absent")
        return False

    cmd = [
        colmap_exe, "model_converter",
        "--input_path=" + str(model_path),
        "--output_path=" + str(output_ply),
        "--output_type=PLY",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode == 0:
        log(f"[OK] PLY exporté: {output_ply}")
        return True
    err = r.stderr if r.stderr else r.stdout
    log(f"[ERROR] Export PLY échoué:\n{err[:1000]}")
    return False


def ensure_text_model_from_sparse(colmap_exe, sparse_dir, txt_dir):
    sparse_dir = Path(sparse_dir)
    txt_dir = Path(txt_dir)
    txt_dir.mkdir(parents=True, exist_ok=True)

    if (txt_dir / "cameras.txt").exists() and (txt_dir / "images.txt").exists() and (txt_dir / "points3D.txt").exists():
        return txt_dir

    cmd = [
        colmap_exe, "model_converter",
        "--input_path=" + str(sparse_dir),
        "--output_path=" + str(txt_dir),
        "--output_type=TXT",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        err = r.stderr if r.stderr else r.stdout
        raise RuntimeError(f"Conversion BIN -> TXT échouée:\n{err[:1200]}")

    return txt_dir


def write_ply_xyzrgb(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None, verbose: int = 1):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz doit être de forme (N, 3), reçu {xyz.shape}")

    finite_mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite_mask]

    if rgb is None:
        rgb = np.full((len(xyz), 3), 200, dtype=np.uint8)
    else:
        rgb = np.asarray(rgb)
        if rgb.ndim != 2 or rgb.shape[1] != 3:
            raise ValueError(f"rgb doit être de forme (N, 3), reçu {rgb.shape}")
        if len(rgb) == len(finite_mask):
            rgb = rgb[finite_mask]
        elif len(rgb) != len(xyz):
            raise ValueError(f"rgb a une longueur incompatible: rgb={len(rgb)}, xyz={len(xyz)}")
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    n = len(xyz)
    if n == 0:
        raise RuntimeError("Aucun point valide à écrire dans sparse_pc.ply")

    iterable = zip(xyz, rgb)
    if verbose and tqdm is not None:
        iterable = tqdm(iterable, total=n, desc="Écriture sparse_pc.ply", unit="pt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in iterable:
            if not np.isfinite(p).all():
                continue
            f.write(
                f"{float(p[0]):.12f} {float(p[1]):.12f} {float(p[2]):.12f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )

# ============================================================================
# PARTIE 5: NORMALISATION + TRANSFORMS.JSON
# ============================================================================

def read_text_model_intrinsics(cameras_txt):
    cameras_txt = Path(cameras_txt)
    for line in cameras_txt.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        toks = s.split()
        if len(toks) < 5:
            continue

        model = toks[1]
        w = int(toks[2])
        h = int(toks[3])
        params = list(map(float, toks[4:]))

        if model == "SIMPLE_PINHOLE":
            f, cx, cy = params[:3]
            fx, fy = f, f
        elif model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model in ("OPENCV", "FULL_OPENCV"):
            fx, fy, cx, cy = params[:4]
        else:
            raise RuntimeError(f"Modèle caméra non supporté pour transforms.json: {model}")

        return {
            "camera_model": model,
            "w": w,
            "h": h,
            "fl_x": fx,
            "fl_y": fy,
            "cx": cx,
            "cy": cy,
        }

    raise RuntimeError(f"Aucune caméra lisible dans {cameras_txt}")


def read_text_images_frames(images_txt, image_subdir="images"):
    images_txt = Path(images_txt)
    lines = images_txt.read_text(encoding="utf-8").splitlines()
    frames = []
    i = 0

    # Conversion de convention caméra: COLMAP (X droite, Y bas, Z avant)
    # vers OpenGL/Nerfstudio (X droite, Y haut, Z arrière).
    # C'est la conversion standard utilisée par les dataparsers COLMAP de
    # Nerfstudio: on inverse les colonnes Y et Z de la rotation c2w, en
    # laissant la colonne X (right) et la translation (center) inchangées.
    CAMERA_AXIS_FLIP = np.diag([1.0, -1.0, -1.0])

    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        toks = line.split()
        if len(toks) < 10:
            i += 1
            continue

        image_id = int(toks[0])
        q = np.array([float(toks[1]), float(toks[2]), float(toks[3]), float(toks[4])], dtype=np.float64)
        t = np.array([float(toks[5]), float(toks[6]), float(toks[7])], dtype=np.float64)
        image_name = " ".join(toks[9:])

        R = quaternion_to_rotmat(q)
        C = translation_to_center(R, t)

        R_c2w = R.T @ CAMERA_AXIS_FLIP

        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = C

        frames.append({
            "file_path": f"./{image_subdir}/{Path(image_name).name}",
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": image_id,
        })
        i += 2

    return frames


def compute_normalization_from_frames(frames):
    centers = np.stack([np.array(f["transform_matrix"], dtype=np.float64)[:3, 3] for f in frames], axis=0)
    center_mean = centers.mean(axis=0)
    deltas = centers - center_mean
    radius_before = float(np.max(np.linalg.norm(deltas, axis=1)))
    scale = 1.0 if radius_before < 1e-12 else 1.0 / radius_before
    radius_after = float(radius_before * scale)
    return center_mean, scale, radius_before, radius_after


def apply_normalization_to_frames(frames, center_mean, scale):
    for f in frames:
        M = np.array(f["transform_matrix"], dtype=np.float64)
        M[:3, 3] = (M[:3, 3] - center_mean) * scale
        f["transform_matrix"] = M.tolist()


def export_georeferencing_sidecars(output_dir, center_mean, scale, radius_before, radius_after):
    output_dir = Path(output_dir)

    normalization = {
        "normalization_type": "translation_then_uniform_scale",
        "translation_subtracted_world_xyz": [float(center_mean[0]), float(center_mean[1]), float(center_mean[2])],
        "uniform_scale_applied": float(scale),
        "camera_radius_before": float(radius_before),
        "camera_radius_after": float(radius_after),
        "forward_formula": "x_normalized = (x_world - translation_subtracted_world_xyz) * uniform_scale_applied",
        "inverse_formula": "x_world = x_normalized / uniform_scale_applied + translation_subtracted_world_xyz",
    }

    georeferencing = {
        "world_frame": "original_colmap_world",
        "normalized_frame": "gaussian_splatting_training_frame",
        "translation_subtracted_world_xyz": [float(center_mean[0]), float(center_mean[1]), float(center_mean[2])],
        "uniform_scale_applied": float(scale),
        "world_to_normalized_4x4": [
            [float(scale), 0.0, 0.0, float(-scale * center_mean[0])],
            [0.0, float(scale), 0.0, float(-scale * center_mean[1])],
            [0.0, 0.0, float(scale), float(-scale * center_mean[2])],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "normalized_to_world_4x4": [
            [float(1.0 / scale if abs(scale) > 1e-15 else 1.0), 0.0, 0.0, float(center_mean[0])],
            [0.0, float(1.0 / scale if abs(scale) > 1e-15 else 1.0), 0.0, float(center_mean[1])],
            [0.0, 0.0, float(1.0 / scale if abs(scale) > 1e-15 else 1.0), float(center_mean[2])],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "units_world": "COLMAP world units (ex: meters if your cartography is metric)",
    }

    normalization_path = output_dir / "transforms_normalization.json"
    georef_path = output_dir / "georeferencing.json"

    with open(normalization_path, "w", encoding="utf-8") as f:
        json.dump(normalization, f, indent=2)

    with open(georef_path, "w", encoding="utf-8") as f:
        json.dump(georeferencing, f, indent=2)

    return normalization_path, georef_path


def export_transforms_json_from_text_model(model_txt_dir, output_json, image_subdir="images", ply_file_path="sparse_pc.ply", normalize=False):
    model_txt_dir = Path(model_txt_dir)
    output_json = Path(output_json)

    cameras_txt = model_txt_dir / "cameras.txt"
    images_txt = model_txt_dir / "images.txt"

    if not cameras_txt.exists() or not images_txt.exists():
        raise RuntimeError(f"Modèle TXT incomplet: {model_txt_dir}")

    intr = read_text_model_intrinsics(cameras_txt)
    frames = read_text_images_frames(images_txt, image_subdir=image_subdir)
    if not frames:
        raise RuntimeError("Aucune image avec pose trouvée pour générer transforms.json")

    ns_camera_model = "OPENCV" if intr["camera_model"] == "FULL_OPENCV" else intr["camera_model"]

    center_mean = np.zeros(3, dtype=np.float64)
    scale = 1.0
    radius_before = 0.0
    radius_after = 0.0

    if normalize:
        center_mean, scale, radius_before, radius_after = compute_normalization_from_frames(frames)
        apply_normalization_to_frames(frames, center_mean, scale)

    transforms = {
        "w": int(intr["w"]),
        "h": int(intr["h"]),
        "fl_x": float(intr["fl_x"]),
        "fl_y": float(intr["fl_y"]),
        "cx": float(intr["cx"]),
        "cy": float(intr["cy"]),
        "camera_model": ns_camera_model,
        "ply_file_path": ply_file_path,
        "frames": frames,
        "applied_transform": [
            [1.0, 0.0, 0.0, float(-center_mean[0])],
            [0.0, 1.0, 0.0, float(-center_mean[1])],
            [0.0, 0.0, 1.0, float(-center_mean[2])],
        ],
        "applied_scale": float(scale),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=2)

    norm_path, georef_path = export_georeferencing_sidecars(
        output_dir=output_json.parent,
        center_mean=center_mean,
        scale=scale,
        radius_before=radius_before,
        radius_after=radius_after,
    )

    log(f"[OK] transforms.json généré: {output_json}")
    log(f"[OK] sidecar normalisation: {norm_path}")
    log(f"[OK] sidecar georeferencing: {georef_path}")
    return center_mean, scale
# ============================================================================
# PARTIE 6: LAZ / FUSION / sparse_pc.ply
# ============================================================================

def apply_transform_to_points(xyz: np.ndarray, T4: np.ndarray, scale: float):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz doit être de forme (N, 3), reçu {xyz.shape}")

    if len(xyz) == 0:
        return xyz.copy()

    T4 = np.asarray(T4, dtype=np.float64)
    if T4.shape != (4, 4):
        raise ValueError(f"T4 doit être de forme (4, 4), reçu {T4.shape}")

    xyz_h = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
    xyz_t = (T4 @ xyz_h.T).T[:, :3]
    xyz_t *= float(scale)
    return xyz_t


def filter_points_by_bbox(
    xyz: np.ndarray,
    rgb: np.ndarray | None = None,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    zmin=None,
    zmax=None,
):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz doit être de forme (N,3), reçu {xyz.shape}")

    mask = np.isfinite(xyz).all(axis=1)

    if xmin is not None:
        mask &= xyz[:, 0] >= float(xmin)
    if xmax is not None:
        mask &= xyz[:, 0] <= float(xmax)
    if ymin is not None:
        mask &= xyz[:, 1] >= float(ymin)
    if ymax is not None:
        mask &= xyz[:, 1] <= float(ymax)
    if zmin is not None:
        mask &= xyz[:, 2] >= float(zmin)
    if zmax is not None:
        mask &= xyz[:, 2] <= float(zmax)

    xyz_out = xyz[mask]
    rgb_out = None if rgb is None else np.asarray(rgb)[mask]
    return xyz_out, rgb_out, mask


def voxel_downsample_xyzrgb(xyz: np.ndarray, rgb: np.ndarray | None, voxel_size: float | None):
    xyz = np.asarray(xyz, dtype=np.float64)
    if voxel_size is None or float(voxel_size) <= 0.0 or len(xyz) == 0:
        return xyz, rgb

    voxel_size = float(voxel_size)
    origin = xyz.min(axis=0)

    voxel_indices = np.floor((xyz - origin) / voxel_size).astype(np.int64)
    _, keep = np.unique(voxel_indices, axis=0, return_index=True)
    keep = np.sort(keep)

    xyz_out = xyz[keep]
    rgb_out = None if rgb is None else np.asarray(rgb)[keep]
    log(f"[LAZ] Voxel downsampling={voxel_size:g} m: {len(xyz)} -> {len(xyz_out)}")
    return xyz_out, rgb_out


def merge_laz_with_colmap_points(colmap_xyz: np.ndarray, laz_xyz: np.ndarray, colmap_rgb=None, laz_rgb=None, max_voxel_dist: float | None = None):
    colmap_xyz = np.asarray(colmap_xyz, dtype=np.float64)
    laz_xyz = np.asarray(laz_xyz, dtype=np.float64)

    if len(colmap_xyz) == 0:
        xyz = laz_xyz.copy()
        rgb = laz_rgb.copy() if laz_rgb is not None else None
        return xyz, rgb

    if len(laz_xyz) == 0:
        xyz = colmap_xyz.copy()
        rgb = colmap_rgb.copy() if colmap_rgb is not None else None
        return xyz, rgb

    if max_voxel_dist is not None and max_voxel_dist > 0.0:
        origin = np.minimum(colmap_xyz.min(axis=0), laz_xyz.min(axis=0))
        step = float(max_voxel_dist)

        def voxelize(arr):
            return np.floor((arr - origin) / step).astype(np.int64)

        cidx = voxelize(colmap_xyz)
        lidx = voxelize(laz_xyz)
        seen = set()
        keep_laz = []

        for i, p in enumerate(laz_xyz):
            key = tuple(lidx[i])
            if key in seen:
                continue
            seen.add(key)
            keep_laz.append(i)

        laz_xyz = laz_xyz[keep_laz]
        if laz_rgb is not None:
            laz_rgb = laz_rgb[keep_laz]

    xyz = np.vstack((colmap_xyz, laz_xyz))
    rgb_parts = []

    if colmap_rgb is not None:
        rgb_parts.append(np.asarray(colmap_rgb))
    if laz_rgb is not None:
        rgb_parts.append(np.asarray(laz_rgb))

    if rgb_parts:
        rgb = np.vstack(rgb_parts)
    else:
        rgb = np.full((len(xyz), 3), 200, dtype=np.uint8)

    xyz, rgb, _ = filter_finite_points(xyz, rgb)
    rgb = np.asarray(rgb, dtype=np.uint8)
    return xyz, rgb


def read_laz_points(laz_path: Path, stride: int = 1):
    if laspy is None:
        raise RuntimeError("Le paramètre --laz requiert laspy. Installe: pip install 'laspy[lazrs]'")

    laz_path = Path(laz_path).resolve()
    if not laz_path.exists():
        raise FileNotFoundError(f"Fichier LAZ introuvable: {laz_path}")

    log(f"[LAZ] Lecture: {laz_path}")
    las = laspy.read(str(laz_path))

    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64, copy=False)

    rgb = None
    if all(hasattr(las, field) for field in ("red", "green", "blue")):
        rgb = np.column_stack((las.red, las.green, las.blue)).astype(np.float64, copy=False)
        rgb_max = float(np.max(rgb)) if len(rgb) else 0.0
        if rgb_max > 255.0:
            rgb = np.round(rgb / 256.0)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 200, dtype=np.uint8)

    xyz, rgb, _ = filter_finite_points(xyz, rgb)

    stride = max(1, int(stride))
    if stride > 1:
        xyz = xyz[::stride]
        rgb = rgb[::stride]

    summarize_points3d(xyz, "laz/source")
    return xyz, rgb


def laz_to_colmap_coordinates(xyz: np.ndarray):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz LAZ doit être (N,3), reçu {xyz.shape}")

    A = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float64)
    out = xyz @ A.T
    summarize_points3d(out, "laz/colmap")
    return out


def prepare_laz_sparse_cloud(args, colmap_xyz: np.ndarray | None = None):
    if not args.laz:
        return None, None

    xyz, rgb = read_laz_points(Path(args.laz), stride=args.laz_stride)
    xyz = laz_to_colmap_coordinates(xyz)

    xyz, rgb, _ = filter_points_by_bbox(
        xyz,
        rgb,
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        zmin=args.zmin,
        zmax=args.zmax,
    )

    if len(xyz) == 0:
        raise RuntimeError("Le filtrage LAZ a supprimé tous les points. Vérifie --xmin/--xmax/--ymin/--ymax/--zmin/--zmax.")

    xyz, rgb = voxel_downsample_xyzrgb(xyz, rgb, args.laz_voxel_size)

    if colmap_xyz is not None and len(colmap_xyz):
        colmap_rgb = np.full((len(colmap_xyz), 3), 200, dtype=np.uint8)
        xyz, rgb = merge_laz_with_colmap_points(
            colmap_xyz=np.asarray(colmap_xyz, dtype=np.float64),
            laz_xyz=xyz,
            colmap_rgb=colmap_rgb,
            laz_rgb=rgb,
            max_voxel_dist=0.05,
        )

    xyz, rgb, _ = filter_finite_points(xyz, rgb)
    if len(xyz) == 0:
        raise RuntimeError("Aucun point valide après fusion LAZ + COLMAP")

    summarize_points3d(xyz, "laz/final_fused_sparse_pc")
    return xyz, rgb


# ============================================================================
# PARTIE 7: MODEL TEXT / BBOX FILTER
# ====================================================================================

def filter_colmap_points3d_by_bbox_text_model(model_dir, xmin, xmax, ymin, ymax, zmin=None, zmax=None, verbose=False):
    pts_path = Path(model_dir) / "points3D.txt"
    if not pts_path.exists():
        raise RuntimeError(f"points3D.txt introuvable: {pts_path}")

    kept, removed, out_lines = 0, 0, []
    with open(pts_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                out_lines.append(line)
                continue
            toks = s.split()
            if len(toks) < 7:
                out_lines.append(line)
                continue

            x, y, z = float(toks[1]), float(toks[2]), float(toks[3])
            in_xy = xmin <= x <= xmax and ymin <= y <= ymax
            in_z = True if (zmin is None or zmax is None) else (zmin <= z <= zmax)

            if in_xy and in_z:
                out_lines.append(line)
                kept += 1
            else:
                removed += 1

    with open(pts_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    v_log(verbose, f"[BBOX] gardés={kept}, supprimés={removed}")
    return kept, removed


# ============================================================================
# PARTIE 8: STATE + CLEANUP
# ============================================================================

def get_state_file(output_dir):
    return Path(output_dir) / ".pipeline_state"


def get_completed_phases(output_dir):
    sf = get_state_file(output_dir)
    if not sf.exists():
        return set()
    with open(sf, "r") as f:
        return set(l.strip() for l in f if l.strip())


def write_completed_phases(output_dir, phases):
    with open(get_state_file(output_dir), "w") as f:
        for p in sorted(phases):
            f.write(p + "\n")


def mark_phase_complete(output_dir, phase):
    done = get_completed_phases(output_dir)
    done.add(phase)
    write_completed_phases(output_dir, done)


def _rm_if_exists(p):
    p = Path(p)
    if p.exists():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


def cleanup_requested_items(output_dir):
    output_dir = Path(output_dir)
    colmap_root = output_dir / "colmap"

    _rm_if_exists(output_dir / "model_georef.ply")
    _rm_if_exists(colmap_root / "sparse_txt_for_transforms")
    _rm_if_exists(colmap_root / "sparse_eval_txt")
    _rm_if_exists(colmap_root / "sparse_init")
    _rm_if_exists(colmap_root / "sparse_txt_for_inspection")


def clear_phase(output_dir, phase_name):
    output_dir = Path(output_dir)
    done = get_completed_phases(output_dir)

    if phase_name == "all":
        _rm_if_exists(get_state_file(output_dir))
        return

    colmap_root = output_dir / "colmap"

    if phase_name == "keypoints":
        for p in ["keypoints", "feature_extraction", "matching", "camera_georeferencing", "triangulation", "export", "georeferencing"]:
            done.discard(p)

        for p in ["transforms.json", "transforms_normalization.json", "georeferencing.json", "sparse_pc.ply", "model.ply", "model_fused.ply"]:
            _rm_if_exists(output_dir / p)

        _rm_if_exists(colmap_root / "database.db")
        _rm_if_exists(colmap_root / "sparse")
        _rm_if_exists(colmap_root / "sparse_bin_bbox")
        _rm_if_exists(colmap_root / "sparse_txt_for_bbox")
        _rm_if_exists(output_dir / "images")
        cleanup_requested_items(output_dir)

    elif phase_name == "matching":
        for p in ["matching", "triangulation", "export", "georeferencing"]:
            done.discard(p)
        for p in ["transforms.json", "transforms_normalization.json", "georeferencing.json", "sparse_pc.ply", "model.ply", "model_fused.ply"]:
            _rm_if_exists(output_dir / p)
        _rm_if_exists(colmap_root / "sparse")
        _rm_if_exists(colmap_root / "sparse_bin_bbox")
        _rm_if_exists(colmap_root / "sparse_txt_for_bbox")
        cleanup_requested_items(output_dir)

    elif phase_name == "triangulation":
        for p in ["camera_georeferencing", "triangulation", "export", "georeferencing"]:
            done.discard(p)
        for p in ["transforms.json", "transforms_normalization.json", "georeferencing.json", "sparse_pc.ply", "model.ply", "model_fused.ply"]:
            _rm_if_exists(output_dir / p)
        _rm_if_exists(colmap_root / "sparse")
        _rm_if_exists(colmap_root / "sparse_bin_bbox")
        _rm_if_exists(colmap_root / "sparse_txt_for_bbox")
        cleanup_requested_items(output_dir)

    elif phase_name == "export":
        for p in ["export", "georeferencing"]:
            done.discard(p)
        for p in ["transforms.json", "transforms_normalization.json", "georeferencing.json", "sparse_pc.ply", "model.ply", "model_fused.ply"]:
            _rm_if_exists(output_dir / p)
        cleanup_requested_items(output_dir)

    else:
        done.discard(phase_name)

    write_completed_phases(output_dir, done)


def reset_all_phases(output_dir):
    output_dir = Path(output_dir)
    _rm_if_exists(get_state_file(output_dir))
    for p in [
        output_dir / "colmap" / "database.db",
        output_dir / "transforms.json",
        output_dir / "transforms_normalization.json",
        output_dir / "georeferencing.json",
        output_dir / "sparse_pc.ply",
        output_dir / "model.ply",
        output_dir / "model_fused.ply",
    ]:
        _rm_if_exists(p)
    _rm_if_exists(output_dir / "colmap" / "sparse")
    _rm_if_exists(output_dir / "colmap" / "sparse_bin_bbox")
    _rm_if_exists(output_dir / "colmap" / "sparse_txt_for_bbox")
    _rm_if_exists(output_dir / "images")
    cleanup_requested_items(output_dir)


# ============================================================================
# PARTIE 9: MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline COLMAP + normalisation + fusion LAZ pour Gaussian Splatting")
    parser.add_argument("-i", "--images", required=True, help="Dossier source contenant JPG et CON")
    parser.add_argument("-o", "--output", required=True, help="Dossier de sortie")
    parser.add_argument("-t", "--tolerance", type=float, default=1.0)
    parser.add_argument("-d", "--detector", choices=["SIFT", "ORB", "AKAZE"], default="SIFT")
    parser.add_argument("-m", "--max-keypoints", type=int, default=None)
    parser.add_argument("--camera-model", choices=["SIMPLE_PINHOLE", "PINHOLE", "OPENCV", "FULL_OPENCV"], default="PINHOLE")
    parser.add_argument("--colmap-exe", default="colmap")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--skip-matching", action="store_true")
    parser.add_argument("--skip-triangulation", action="store_true")
    parser.add_argument("--skip-bundle-adjustment", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--clean", choices=["all", "keypoints", "matching", "triangulation", "export"])
    parser.add_argument("--xmin", type=float, default=None)
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--zmin", type=float, default=None)
    parser.add_argument("--zmax", type=float, default=None)
    parser.add_argument("--laz", default=None, help="Fichier LAS/LAZ optionnel. Fusionné avec COLMAP pour sparse_pc.ply.")
    parser.add_argument("--laz-stride", type=int, default=1, help="Sous-échantillonnage du LAZ (1 = entier).")
    parser.add_argument("--laz-voxel-size", type=float, default=0.5, help="Voxel de fusion LAZ en mètres. Défault 0.5.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    images_src_dir = Path(args.images).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.laz_stride < 1:
        raise ValueError("--laz-stride doit être >= 1")
    if args.laz and laspy is None:
        raise RuntimeError("--laz a été fourni mais laspy est absent. Installe: pip install 'laspy[lazrs]'")

    laz_xyz_for_sparse_pc = None
    laz_rgb_for_sparse_pc = None

    if not images_src_dir.exists():
        raise RuntimeError(f"Dossier images inexistant: {images_src_dir}")

    if args.clean:
        log(f"\n[CLEAN] Nettoyage phase: {args.clean}")
        if args.clean == "all":
            reset_all_phases(output_dir)
        else:
            clear_phase(output_dir, args.clean)
        log("[OK] Nettoyage terminé")
        return True

    completed = get_completed_phases(output_dir)
    output_images_dir = prepare_output_images_dir(images_src_dir, output_dir, verbose=args.verbose)

    colmap_root = output_dir / "colmap"
    colmap_root.mkdir(parents=True, exist_ok=True)

    db_path = colmap_root / "database.db"
    sparse_dir = resolve_sparse_dir(output_dir) if ((colmap_root / "sparse" / "0").exists() or (output_dir / "sparse" / "0").exists()) else (colmap_root / "sparse" / "0")
    sparse_init_dir = colmap_root / "sparse_init" / "0"

    if args.verbose:
        log("\n" + "=" * 72)
        log("PIPELINE MATIS + INJECTION POSES .CON EN AMONT")
        log("=" * 72)
        log(f"Images source       : {images_src_dir}")
        log(f"Images utilisées    : {output_images_dir}")
        log(f"Output              : {output_dir}")
        log(f"COLMAP root         : {colmap_root}")
        log(f"Phases complétées   : {completed or 'aucune'}")

    records = []
    jpgs = sorted(output_images_dir.glob("*.jpg")) + sorted(output_images_dir.glob("*.JPG")) + sorted(output_images_dir.glob("*.jpeg")) + sorted(output_images_dir.glob("*.JPEG"))
    if not jpgs:
        raise RuntimeError(f"Aucune image dans {output_images_dir}")

    if "keypoints" not in completed:
        log("\n" + "=" * 70)
        log("PHASE 1 : LECTURE .CON + STATS KEYPOINTS")
        log("=" * 70)

        for idx, jpg in enumerate(jpgs, start=1):
            con = _find_con_for_image(output_images_dir, jpg.stem)
            if con is None:
                v_log(args.verbose, f"[{idx}/{len(jpgs)}] [SKIP] CON absent: {jpg.stem}.CON/.con")
                continue

            rec = read_con(con, jpg_filename=jpg.name)
            records.append(rec)

            v_log(args.verbose, f"[{idx}/{len(jpgs)}] [OK] {jpg.name} | C={rec['center']}")
            ensure_finite_array(f"CON center {jpg.name}", rec["center"])
            ensure_finite_array(f"CON R {jpg.name}", rec["R"])

            try:
                kp, _ = extract_keypoints(jpg, args.detector, args.max_keypoints)
                filter_keypoints_by_tolerance(kp, rec, jpg, verbose=args.verbose)
            except Exception as e:
                v_log(args.verbose, f"[WARN] keypoints {jpg.name}: {e}")

        if len(records) < 3:
            raise RuntimeError("Pas assez de couples JPG/CON valides (<3).")

        first = records[0]
        for rec in records[1:]:
            if (
                rec["width"] != first["width"]
                or rec["height"] != first["height"]
                or not np.isclose(rec["focal"], first["focal"])
                or not np.isclose(rec["cx"], first["cx"])
                or not np.isclose(rec["cy"], first["cy"])
            ):
                raise RuntimeError("Intrinsèques non identiques entre images.")

        centers = np.array([r["center"] for r in records], dtype=np.float64)
        log(f"[INFO] {len(records)} images .CON valides")
        log(f"[INFO] Centroïde: {centers.mean(axis=0)}")

        log("\n" + "=" * 70)
        log("PHASE 2 : CREATION DATABASE")
        log("=" * 70)
        create_colmap_database(args.colmap_exe, db_path)
        mark_phase_complete(output_dir, "keypoints")
        completed = get_completed_phases(output_dir)
    else:
        log("\n[SKIP] Phase keypoints déjà complétée")
        for jpg in jpgs:
            con = _find_con_for_image(output_images_dir, jpg.stem)
            if con is not None:
                try:
                    records.append(read_con(con, jpg_filename=jpg.name))
                except Exception:
                    pass
        if len(records) < 3:
            raise RuntimeError("Pas assez de .CON rechargeables.")

    if args.laz:
        log("\n" + "=" * 70)
        log("PREPARATION DU NUAGE LAZ POUR LE FUSIONNEMENT")
        log("=" * 70)

        colmap_xyz_for_sparse = None
        try:
            txt_for_sparse = output_dir / "colmap" / "sparse_txt_for_inspection"
            if txt_for_sparse.exists() and (txt_for_sparse / "points3D.txt").exists():
                colmap_xyz_for_sparse = read_points3d_txt(txt_for_sparse / "points3D.txt")
        except Exception:
            colmap_xyz_for_sparse = None

        laz_xyz_for_sparse_pc, laz_rgb_for_sparse_pc = prepare_laz_sparse_cloud(
            args,
            colmap_xyz=colmap_xyz_for_sparse,
        )
        log(f"[LAZ] sparse_pc.ply sera construit depuis un nuage fusionné ({len(laz_xyz_for_sparse_pc)} points)")
    else:
        log("[INFO] Aucun --laz; sparse_pc.ply sera construit comme avant.")

    if not check_colmap_installed(args.colmap_exe):
        raise RuntimeError("COLMAP introuvable.")

    log("\n" + "=" * 70)
    log("PHASE 3 : COLMAP")
    log("=" * 70)

    if "feature_extraction" not in completed:
        cmd = [
            args.colmap_exe, "feature_extractor",
            "--database_path=" + str(db_path),
            "--image_path=" + str(output_images_dir),
            "--ImageReader.single_camera=1",
            "--ImageReader.camera_model=" + args.camera_model,
            "--SiftExtraction.use_gpu=1",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            err = r.stderr if r.stderr else r.stdout
            raise RuntimeError(f"feature_extractor échoué:\n{err[:1200]}")

        add_camera_to_database(db_path, records, camera_model=args.camera_model)
        mark_phase_complete(output_dir, "feature_extraction")
        completed = get_completed_phases(output_dir)
        log("[OK] Feature extraction")
    else:
        log("[SKIP] feature_extraction")

    if "camera_georeferencing" not in completed:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT image_id, name FROM images ORDER BY image_id")
        rows = cur.fetchall()
        conn.close()

        image_name_to_id = {Path(name).name: image_id for image_id, name in rows}
        n_written = write_init_model_from_con(records, image_name_to_id, sparse_init_dir, args.camera_model)
        log(f"[OK] Modèle init écrit ({n_written} poses)")
        log_pose_residuals_against_con(records, sparse_init_dir / "images.txt", verbose=args.verbose)
        mark_phase_complete(output_dir, "camera_georeferencing")
        completed = get_completed_phases(output_dir)
    else:
        log("[SKIP] camera_georeferencing")

    ensure_init_model_exists_from_db_and_con(db_path, records, sparse_init_dir, args.camera_model)

    if not args.skip_matching and "matching" not in completed:
        if not run_colmap_feature_matching(args.colmap_exe, db_path, gpu=not args.no_gpu):
            return False
        mark_phase_complete(output_dir, "matching")
        completed = get_completed_phases(output_dir)

    if not args.skip_triangulation and "triangulation" not in completed:
        ok = run_colmap_point_triangulator(args.colmap_exe, db_path, output_images_dir, sparse_init_dir, sparse_dir)
        if not ok:
            return False
        mark_phase_complete(output_dir, "triangulation")
        completed = get_completed_phases(output_dir)

        txt_inspect = colmap_root / "sparse_txt_for_inspection"
        _rm_if_exists(txt_inspect)
        try:
            ensure_text_model_from_sparse(args.colmap_exe, sparse_dir, txt_inspect)
            if (txt_inspect / "points3D.txt").exists():
                tri_pts = read_points3d_txt(txt_inspect / "points3D.txt")
                summarize_points3d(tri_pts, "triangulation/points3D.txt")
        except Exception as e:
            log(f"[POINTS3D][triangulation][WARN] conversion TXT impossible: {e}")

    elif "triangulation" in completed:
        log("[SKIP] triangulation")

    have_bbox = all(v is not None for v in [args.xmin, args.xmax, args.ymin, args.ymax])
    if "triangulation" in get_completed_phases(output_dir) and have_bbox:
        txt_dir = colmap_root / "sparse_txt_for_bbox"
        bin_dir = colmap_root / "sparse_bin_bbox"
        _rm_if_exists(txt_dir)
        _rm_if_exists(bin_dir)
        txt_dir.mkdir(parents=True, exist_ok=True)
        bin_dir.mkdir(parents=True, exist_ok=True)

        run_model_converter(args.colmap_exe, sparse_dir, txt_dir, "TXT")
        if (txt_dir / "points3D.txt").exists():
            bbox_pts = read_points3d_txt(txt_dir / "points3D.txt")
            summarize_points3d(bbox_pts, "bbox/input_points3D.txt")

        filter_colmap_points3d_by_bbox_text_model(
            txt_dir, args.xmin, args.xmax, args.ymin, args.ymax, args.zmin, args.zmax, verbose=args.verbose
        )

        if (txt_dir / "points3D.txt").exists():
            bbox_pts_after = read_points3d_txt(txt_dir / "points3D.txt")
            summarize_points3d(bbox_pts_after, "bbox/filtered_points3D.txt")

        run_model_converter(args.colmap_exe, txt_dir, bin_dir, "BIN")

        sparse_dir.mkdir(parents=True, exist_ok=True)
        for f in sparse_dir.glob("*"):
            _rm_if_exists(f)
        for f in bin_dir.glob("*"):
            dst = sparse_dir / f.name
            if f.is_file():
                shutil.copy2(f, dst)
            else:
                shutil.copytree(f, dst)
        log("[OK] Filtrage bbox appliqué")

    if not args.skip_bundle_adjustment and "triangulation" in get_completed_phases(output_dir):
        run_colmap_bundle_adjustment(args.colmap_exe, sparse_dir)

    if (sparse_dir / "images.bin").exists():
        eval_txt = colmap_root / "sparse_eval_txt"
        _rm_if_exists(eval_txt)
        eval_txt.mkdir(parents=True, exist_ok=True)
        try:
            run_model_converter(args.colmap_exe, sparse_dir, eval_txt, "TXT")
            if (eval_txt / "images.txt").exists():
                log_pose_residuals_against_con(records, eval_txt / "images.txt", verbose=args.verbose)
        except Exception as e:
            log(f"[WARNING] Eval résidus impossible: {e}")

    transforms_path = output_dir / "transforms.json"
    center_mean = np.zeros(3, dtype=np.float64)
    scale = 1.0

    if (sparse_dir / "images.bin").exists() or (sparse_dir / "images.txt").exists():
        txt_for_transforms = colmap_root / "sparse_txt_for_transforms"
        _rm_if_exists(txt_for_transforms)
        txt_for_transforms.mkdir(parents=True, exist_ok=True)

        try:
            run_model_converter(args.colmap_exe, sparse_dir, txt_for_transforms, "TXT")
            center_mean, scale = export_transforms_json_from_text_model(
                model_txt_dir=txt_for_transforms,
                output_json=transforms_path,
                image_subdir="images",
                ply_file_path="sparse_pc.ply",
                normalize=False,
            )
        except Exception as e:
            log(f"[WARNING] Impossible de générer transforms.json: {e}")
    else:
        log("[SKIP] transforms.json (modèle incomplet)")

    output_ply = output_dir / "model.ply"
    if not args.skip_export and (sparse_dir / "cameras.bin").exists():
        if export_model_ply(args.colmap_exe, sparse_dir, output_ply):
            mark_phase_complete(output_dir, "export")
            completed = get_completed_phases(output_dir)

    sparse_pc = output_dir / "sparse_pc.ply"
    model_fused = output_dir / "model_fused.ply"

    if laz_xyz_for_sparse_pc is not None:
        log("\n" + "=" * 70)
        log("[LAZ] ECRITURE sparse_pc.ply DEPUIS LE NUAGE FUSIONNE")
        log("=" * 70)

        write_ply_xyzrgb(
            sparse_pc,
            laz_xyz_for_sparse_pc,
            rgb=laz_rgb_for_sparse_pc,
            verbose=args.verbose,
        )

        stats = ply_summary(sparse_pc, label="sparse_pc_fused_laz", show_bad=20)
        if stats["n_valid"] == 0:
            raise RuntimeError("sparse_pc.ply fusionné vide après écriture")

        log(f"[OK] sparse_pc.ply créé depuis fusion COLMAP + LAZ: {sparse_pc}")
    else:
        source_ply = None
        if model_fused.exists():
            source_ply = model_fused
        elif output_ply.exists():
            source_ply = output_ply

        if source_ply is not None:
            stats = ply_summary(source_ply, label="source_ply", show_bad=20)

            if stats["n_invalid"] > 0:
                log(f"[WARN] {stats['n_invalid']} point(s) invalides détectés dans {source_ply.name}")
                log("[WARN] Nettoyage automatique du PLY avant export sparse_pc.ply")

                pts = read_ply_points(source_ply)
                finite_mask = np.isfinite(pts).all(axis=1)
                pts_clean = pts[finite_mask]

                if len(pts_clean) == 0:
                    raise RuntimeError("Tous les points du PLY sont invalides après filtrage")

                write_ply_xyzrgb(sparse_pc, pts_clean, rgb=None, verbose=args.verbose)
                log(f"[OK] sparse_pc.ply nettoyé et recréé depuis {source_ply.name}: {sparse_pc}")
            else:
                shutil.copy2(source_ply, sparse_pc)
                log(f"[OK] sparse_pc.ply créé depuis {source_ply.name}: {sparse_pc}")
        else:
            log("[WARNING] sparse_pc.ply non créé (ni model_fused.ply ni model.ply)")

    cleanup_requested_items(output_dir)

    log("\n" + "=" * 70)
    log("PIPELINE TERMINÉ")
    log("=" * 70)
    log(f"Images output      : {output_images_dir}")
    log(f"Database COLMAP    : {db_path}")
    log(f"Sparse model       : {sparse_dir}")
    log(f"Transforms         : {output_dir / 'transforms.json'}")
    log(f"Normalisation      : {output_dir / 'transforms_normalization.json'}")
    log(f"Georeferencing     : {output_dir / 'georeferencing.json'}")
    log(f"PLY référence      : {output_dir / 'sparse_pc.ply'}")
    if args.laz:
        log(f"LAZ source         : {Path(args.laz).resolve()}")
        log(f"LAZ stride         : {args.laz_stride}")
        log(f"LAZ voxel size     : {args.laz_voxel_size}")
        log("Initialisation NS  : sparse_pc.ply issu du nuage fusionné COLMAP + LAZ")
    else:
        log("Initialisation NS  : sparse_pc.ply issu de COLMAP")
    return True


# ============================================================================
# PLY SUMMARY
# ============================================================================

def ply_summary(path: Path, label: str = "PLY", show_bad: int = 10):
    path = Path(path)
    if not path.exists():
        log(f"[PLY][{label}] fichier absent: {path}")
        return {
            "exists": False,
            "format": None,
            "n_verts": 0,
            "n_valid": 0,
            "n_invalid": 0,
            "bbox_min": None,
            "bbox_max": None,
            "bbox_diag": None,
            "mean": None,
            "bad_indices": [],
        }

    with open(path, "rb") as f:
        fmt = None
        n_verts = None
        vertex_props = []
        in_vertex_block = False

        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"{path}: header PLY tronqué")
            s = line.decode("ascii", errors="ignore").strip()

            if s.startswith("format "):
                fmt = s.split()[1]
            elif s.startswith("element vertex "):
                n_verts = int(s.split()[-1])
                in_vertex_block = True
            elif s.startswith("element ") and not s.startswith("element vertex "):
                in_vertex_block = False
            elif s.startswith("property ") and in_vertex_block:
                toks = s.split()
                if len(toks) >= 3:
                    vertex_props.append((toks[1], toks[2]))
            elif s == "end_header":
                break

        if fmt is None or n_verts is None:
            raise RuntimeError(f"{path}: format ou element vertex manquant")

        if fmt == "ascii":
            pts = []
            bad_indices = []
            for i in range(n_verts):
                line = f.readline()
                if not line:
                    bad_indices.append(i)
                    break
                toks = line.decode("ascii", errors="ignore").split()
                if len(toks) < 3:
                    bad_indices.append(i)
                    continue
                try:
                    x, y, z = float(toks[0]), float(toks[1]), float(toks[2])
                    pts.append([x, y, z])
                except Exception:
                    bad_indices.append(i)
                    continue
            pts = np.array(pts, dtype=np.float64) if pts else np.zeros((0, 3), dtype=np.float64)
            finite_mask = np.isfinite(pts).all(axis=1)
            valid = pts[finite_mask]
            invalid = len(pts) - len(valid)
            bad_indices = [i for i, bad in enumerate(~finite_mask) if bad]
        else:
            endian = "<" if fmt == "binary_little_endian" else ">"
            ply_to_struct = {
                "char": "b", "int8": "b",
                "uchar": "B", "uint8": "B",
                "short": "h", "int16": "h",
                "ushort": "H", "uint16": "H",
                "int": "i", "int32": "i",
                "uint": "I", "uint32": "I",
                "float": "f", "float32": "f",
                "double": "d", "float64": "d",
            }

            fmt_chars = []
            x_idx = y_idx = z_idx = None
            for idx, (ptype, pname) in enumerate(vertex_props):
                if ptype not in ply_to_struct:
                    raise RuntimeError(f"{path}: type PLY non supporté: {ptype}")
                fmt_chars.append(ply_to_struct[ptype])
                if pname == "x":
                    x_idx = idx
                elif pname == "y":
                    y_idx = idx
                elif pname == "z":
                    z_idx = idx

            if x_idx is None or y_idx is None or z_idx is None:
                raise RuntimeError(f"{path}: propriétés x/y/z manquantes dans le header")

            vertex_struct = struct.Struct(endian + "".join(fmt_chars))
            vertex_size = vertex_struct.size

            pts = []
            bad_indices = []
            for i in range(n_verts):
                raw = f.read(vertex_size)
                if len(raw) < vertex_size:
                    bad_indices.append(i)
                    break
                try:
                    vals = vertex_struct.unpack(raw)
                    x, y, z = vals[x_idx], vals[y_idx], vals[z_idx]
                    pts.append([x, y, z])
                except Exception:
                    bad_indices.append(i)
                    continue

            pts = np.array(pts, dtype=np.float64) if pts else np.zeros((0, 3), dtype=np.float64)
            finite_mask = np.isfinite(pts).all(axis=1)
            valid = pts[finite_mask]
            invalid = len(pts) - len(valid)
            bad_indices = [i for i, bad in enumerate(~finite_mask) if bad]

    if len(valid) == 0:
        log(f"[PLY][{label}] total={len(pts)} valid=0 invalid={invalid}")
        return {
            "exists": True,
            "format": fmt,
            "n_verts": n_verts,
            "n_valid": 0,
            "n_invalid": invalid,
            "bbox_min": None,
            "bbox_max": None,
            "bbox_diag": None,
            "mean": None,
            "bad_indices": bad_indices[:show_bad],
        }

    mn = valid.min(axis=0)
    mx = valid.max(axis=0)
    diag = float(np.linalg.norm(mx - mn))
    mean = valid.mean(axis=0)

    log(f"[PLY][{label}] total={len(pts)} valid={len(valid)} invalid={invalid}")
    log(f"[PLY][{label}] bbox min={mn}")
    log(f"[PLY][{label}] bbox max={mx}")
    log(f"[PLY][{label}] bbox diag={diag:.6f}")
    log(f"[PLY][{label}] mean={mean}")

    if invalid > 0:
        log(f"[PLY][{label}] premiers indices invalides: {bad_indices[:show_bad]}")

    return {
        "exists": True,
        "format": fmt,
        "n_verts": n_verts,
        "n_valid": len(valid),
        "n_invalid": invalid,
        "bbox_min": mn,
        "bbox_max": mx,
        "bbox_diag": diag,
        "mean": mean,
        "bad_indices": bad_indices[:show_bad],
    }


if __name__ == "__main__":
    try:
        ok = main()
        raise SystemExit(0 if ok else 1)
    except Exception as e:
        log(f"\n[FATAL] {e}")
        raise SystemExit(1)
        
        

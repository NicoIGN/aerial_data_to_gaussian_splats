#!/usr/bin/env python3

import argparse
import math
import shutil
import sqlite3
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


# ============================================================================
# LOG
# ============================================================================

def log(msg):
    print(msg, flush=True)


def v_log(verbose, msg):
    if verbose:
        print(msg, flush=True)


# ============================================================================
# PARTIE 1 : LECTURE .CON
# ============================================================================

def read_con(path, jpg_filename=None):
    root = ET.parse(path).getroot()

    def val(xpath, default=None):
        node = root.find(xpath)
        if node is None or node.text is None:
            return default
        return float(node.text)

    image_name = jpg_filename if jpg_filename is not None else (root.findtext(".//image_name") or path.stem)

    center = np.array([
        val(".//extrinseque/systeme/euclidien/x"),
        val(".//extrinseque/systeme/euclidien/y"),
        val(".//extrinseque/sommet/altitude"),
    ], dtype=np.float64)

    R = np.array([
        [val(".//rotation/mat3d/l1/pt3d/x"), val(".//rotation/mat3d/l1/pt3d/y"), val(".//rotation/mat3d/l1/pt3d/z")],
        [val(".//rotation/mat3d/l2/pt3d/x"), val(".//rotation/mat3d/l2/pt3d/y"), val(".//rotation/mat3d/l2/pt3d/z")],
        [val(".//rotation/mat3d/l3/pt3d/x"), val(".//rotation/mat3d/l3/pt3d/y"), val(".//rotation/mat3d/l3/pt3d/z")],
    ], dtype=np.float64)

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


# ============================================================================
# PARTIE 2 : GEOMETRIE
# ============================================================================

def rotmat_to_quaternion(R):
    trace = np.trace(R)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q


def quaternion_to_rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def camera_center_to_translation(R, C):
    return -R @ C


def translation_to_center(R, t):
    return -R.T @ t


# ============================================================================
# PARTIE 3 : PREP IMAGES + FILTRES
# ============================================================================

def prepare_output_images_dir(src_images_dir, output_dir, verbose=False):
    src_images_dir = Path(src_images_dir).resolve()
    out_images = Path(output_dir).resolve() / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    files = []
    for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
        files.extend(src_images_dir.glob(ext))
    files = sorted(files)

    if not files:
        raise RuntimeError(f"Aucune image trouvée dans {src_images_dir}")

    copied = 0
    for f in files:
        shutil.copy2(f, out_images / f.name)
        copied += 1

    v_log(verbose, f"[IMAGES] Copiées: {copied} -> {out_images}")
    return out_images


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

    v_log(verbose, f"    Filtrage: {len(keypoints)} -> {len(filtered_kp)} ({100*len(filtered_kp)/max(1,len(keypoints)):.1f}%)")
    return filtered_kp, filtered_idx


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
# PARTIE 4 : DB / ETAT
# ============================================================================

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


def colmap_model_id(camera_model):
    return {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "OPENCV": 4, "FULL_OPENCV": 6}[camera_model]


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


def clear_phase(output_dir, phase_name):
    output_dir = Path(output_dir)
    done = get_completed_phases(output_dir)

    if phase_name == "all":
        reset_all_phases(output_dir)
        return

    if phase_name == "keypoints":
        for p in ["keypoints", "feature_extraction", "matching", "camera_georeferencing", "triangulation", "export", "georeferencing"]:
            done.discard(p)
        for p in ["colmap/database.db", "model.ply", "model_georef.ply", "sparse_pc.ply", "model_fused.ply"]:
            _rm_if_exists(output_dir / p)
        for d in ["colmap/sparse", "colmap/sparse_init", "colmap/sparse_txt_for_bbox", "colmap/sparse_bin_bbox", "colmap/sparse_eval_txt", "images"]:
            _rm_if_exists(output_dir / d)

    elif phase_name == "matching":
        for p in ["matching", "triangulation", "export", "georeferencing"]:
            done.discard(p)
        for p in ["model.ply", "model_georef.ply", "sparse_pc.ply", "model_fused.ply"]:
            _rm_if_exists(output_dir / p)
        for d in ["colmap/sparse", "colmap/sparse_txt_for_bbox", "colmap/sparse_bin_bbox", "colmap/sparse_eval_txt"]:
            _rm_if_exists(output_dir / d)

    elif phase_name == "triangulation":
        for p in ["camera_georeferencing", "triangulation", "export", "georeferencing"]:
            done.discard(p)
        for p in ["model.ply", "model_georef.ply", "sparse_pc.ply", "model_fused.ply"]:
            _rm_if_exists(output_dir / p)
        for d in ["colmap/sparse", "colmap/sparse_init", "colmap/sparse_txt_for_bbox", "colmap/sparse_bin_bbox", "colmap/sparse_eval_txt"]:
            _rm_if_exists(output_dir / d)

    elif phase_name == "export":
        for p in ["export", "georeferencing"]:
            done.discard(p)
        for p in ["model.ply", "model_georef.ply", "sparse_pc.ply", "model_fused.ply"]:
            _rm_if_exists(output_dir / p)

    else:
        done.discard(phase_name)

    write_completed_phases(output_dir, done)


def reset_all_phases(output_dir):
    output_dir = Path(output_dir)
    _rm_if_exists(get_state_file(output_dir))
    for p in ["colmap/database.db", "model.ply", "model_georef.ply", "sparse_pc.ply", "model_fused.ply"]:
        _rm_if_exists(output_dir / p)
    for d in ["colmap/sparse", "colmap/sparse_init", "colmap/sparse_txt_for_bbox", "colmap/sparse_bin_bbox", "colmap/sparse_eval_txt", "images"]:
        _rm_if_exists(output_dir / d)


# ============================================================================
# PARTIE 5 : COLMAP RUNNERS
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
    cmd = [
        colmap_exe, "model_converter",
        "--input_path=" + str(input_path),
        "--output_path=" + str(output_path),
        "--output_type=" + output_type,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        err = r.stderr if r.stderr else r.stdout
        raise RuntimeError(f"model_converter échoué ({output_type}):\n{err[:1200]}")


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

    if not (input_model_dir / "images.txt").exists():
        log(f"[ERROR] input_model_dir invalide: {input_model_dir}")
        return False

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


# ============================================================================
# PARTIE 6 : MODELE INIT + RESIDUS
# ============================================================================

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
# PARTIE 7 : SORTIE REFERENCE
# ============================================================================

def finalize_sparse_pc_ply(output_dir, verbose=False):
    output_dir = Path(output_dir).resolve()
    fused = output_dir / "model_fused.ply"
    model = output_dir / "model.ply"
    sparse_pc = output_dir / "sparse_pc.ply"

    if fused.exists():
        shutil.copy2(fused, sparse_pc)
        v_log(verbose, f"[PLY] sparse_pc.ply <- {fused.name}")
        return True

    if model.exists():
        shutil.copy2(model, sparse_pc)
        v_log(verbose, f"[PLY] sparse_pc.ply <- {model.name}")
        return True

    return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline COLMAP avec poses .CON injectées en amont")

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
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    images_src_dir = Path(args.images).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Organisation imposée
    output_images_dir = prepare_output_images_dir(images_src_dir, output_dir, verbose=args.verbose)
    colmap_root = output_dir / "colmap"
    colmap_root.mkdir(parents=True, exist_ok=True)

    db_path = colmap_root / "database.db"
    sparse_dir = colmap_root / "sparse" / "0"
    sparse_init_dir = colmap_root / "sparse_init" / "0"

    completed = get_completed_phases(output_dir)

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
    jpgs = sorted(output_images_dir.glob("*.jpg")) + sorted(output_images_dir.glob("*.JPG")) + \
           sorted(output_images_dir.glob("*.jpeg")) + sorted(output_images_dir.glob("*.JPEG"))
    if not jpgs:
        raise RuntimeError(f"Aucune image dans {output_images_dir}")

    # ---- PHASE 1 ----
    if "keypoints" not in completed:
        log("\n" + "=" * 70)
        log("PHASE 1 : LECTURE .CON + STATS KEYPOINTS")
        log("=" * 70)

        for idx, jpg in enumerate(jpgs, start=1):
            con = images_src_dir / (jpg.stem + ".CON")
            if not con.exists():
                v_log(args.verbose, f"[{idx}/{len(jpgs)}] [SKIP] CON absent: {con.name}")
                continue

            rec = read_con(con, jpg_filename=jpg.name)
            records.append(rec)

            v_log(args.verbose, f"[{idx}/{len(jpgs)}] [OK] {jpg.name} | C={rec['center']}")

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
            con = images_src_dir / (jpg.stem + ".CON")
            if con.exists():
                try:
                    records.append(read_con(con, jpg_filename=jpg.name))
                except Exception:
                    pass
        if len(records) < 3:
            raise RuntimeError("Pas assez de .CON rechargeables.")

    # ---- PHASE 3 ----
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
    elif "matching" in completed:
        log("[SKIP] matching")
    else:
        log("[SKIP] matching désactivé")

    if not args.skip_triangulation and "triangulation" not in completed:
        ok = run_colmap_point_triangulator(args.colmap_exe, db_path, output_images_dir, sparse_init_dir, sparse_dir)
        if not ok:
            return False
        mark_phase_complete(output_dir, "triangulation")
        completed = get_completed_phases(output_dir)
    elif "triangulation" in completed:
        log("[SKIP] triangulation")
    else:
        log("[SKIP] triangulation désactivée")

    # bbox filter points3D
    have_bbox = all(v is not None for v in [args.xmin, args.xmax, args.ymin, args.ymax])
    if "triangulation" in completed and have_bbox:
        txt_dir = colmap_root / "sparse_txt_for_bbox"
        bin_dir = colmap_root / "sparse_bin_bbox"
        _rm_if_exists(txt_dir)
        _rm_if_exists(bin_dir)
        txt_dir.mkdir(parents=True, exist_ok=True)
        bin_dir.mkdir(parents=True, exist_ok=True)

        run_model_converter(args.colmap_exe, sparse_dir, txt_dir, "TXT")
        filter_colmap_points3d_by_bbox_text_model(
            txt_dir, args.xmin, args.xmax, args.ymin, args.ymax, args.zmin, args.zmax, verbose=args.verbose
        )
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
    elif "triangulation" in completed:
        log("[BBOX] ignoré (xmin/xmax/ymin/ymax manquants)")

    if not args.skip_bundle_adjustment and "triangulation" in completed:
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

    # export PLY
    output_ply = output_dir / "model.ply"
    if not args.skip_export and (sparse_dir / "cameras.bin").exists():
        if export_model_ply(args.colmap_exe, sparse_dir, output_ply):
            mark_phase_complete(output_dir, "export")
            completed = get_completed_phases(output_dir)

    # compat georef no-op
    if "triangulation" in completed and "georeferencing" not in completed:
        if output_ply.exists():
            shutil.copy2(output_ply, output_dir / "model_georef.ply")
        mark_phase_complete(output_dir, "georeferencing")
        completed = get_completed_phases(output_dir)

    if finalize_sparse_pc_ply(output_dir, verbose=args.verbose):
        log(f"[OK] PLY référence: {output_dir / 'sparse_pc.ply'}")
    else:
        log("[WARNING] Impossible de produire sparse_pc.ply (ni model_fused.ply ni model.ply)")

    log("\n" + "=" * 70)
    log("PIPELINE TERMINÉ")
    log("=" * 70)
    log(f"Images output      : {output_images_dir}")
    log(f"Database COLMAP    : {db_path}")
    log(f"Sparse model       : {sparse_dir}")
    log(f"Init model .CON    : {sparse_init_dir}")
    log(f"PLY référence      : {output_dir / 'sparse_pc.ply'}")
    return True


if __name__ == "__main__":
    try:
        ok = main()
        raise SystemExit(0 if ok else 1)
    except Exception as e:
        log(f"\n[FATAL] {e}")
        raise SystemExit(1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
import sys
from datetime import datetime
import time
from pathlib import Path
import subprocess
import re
import math
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

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

try:
    from PIL import Image, ExifTags
except ImportError:
    print("Erreur: Pillow requis. Installe: pip install pillow")
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


def load_eors(path: Path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()

    if not first.startswith("#"):
        raise RuntimeError(f"Header attendu commençant par '#', trouvé: {first}")

    header = first[1:].strip()
    header_cols = re.split(r"\s+", header)

    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        comment="#",
        header=None,
        names=header_cols,
    )

    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "pointlabel":
            rename[c] = "label"
        elif cl == "x0":
            rename[c] = "X0"
        elif cl == "y0":
            rename[c] = "Y0"
        elif cl == "z0":
            rename[c] = "Z0"
        elif cl.startswith("omega"):
            rename[c] = "omega_deg"
        elif cl.startswith("phi"):
            rename[c] = "phi_deg"
        elif cl.startswith("kappa"):
            rename[c] = "kappa_deg"

    df = df.rename(columns=rename)
    required = ["label", "X0", "Y0", "Z0", "omega_deg", "phi_deg", "kappa_deg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Colonnes manquantes dans {path}: {missing}")

    return df


def load_tp3d(tp_path: Path):
    pts = {}
    with open(tp_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.replace("\t", " ").split()
            if len(p) < 4:
                continue
            try:
                pid = int(float(p[0]))
                pts[pid] = (float(p[1]), float(p[2]), float(p[3]))
            except Exception:
                pass
    return pts


def parse_obs_file(obs_path: Path, max_points_per_image=100):
    rows = []
    with open(obs_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.replace("\t", " ").split()
            if len(p) < 3:
                continue
            try:
                pid = int(float(p[0]))
                c = float(p[1])
                l = float(p[2])
                rows.append((pid, c, l))
            except Exception:
                pass

    if max_points_per_image > 0 and len(rows) > max_points_per_image:
        idx = np.linspace(0, len(rows) - 1, num=max_points_per_image, dtype=int)
        rows = [rows[i] for i in idx]

    return rows


def _exif_dict(img):
    exif = img.getexif()
    if not exif:
        return {}
    out = {}
    for tag_id, val in exif.items():
        name = ExifTags.TAGS.get(tag_id, tag_id)
        out[name] = val
    return out


def _rational_to_float(v):
    try:
        if hasattr(v, "numerator") and hasattr(v, "denominator"):
            return float(v.numerator) / float(v.denominator)
        if isinstance(v, tuple) and len(v) == 2 and v[1] != 0:
            return float(v[0]) / float(v[1])
        return float(v)
    except Exception:
        return None


def infer_image_size_and_initial_f(images_dir: Path, pixel_size_m=4.52e-6):
    candidates = []
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
            candidates.append(p)

    if not candidates:
        raise RuntimeError(f"Aucune image lisible trouvée dans {images_dir}")

    img_path = candidates[0]
    with Image.open(img_path) as img:
        width, height = img.size
        exif = _exif_dict(img)

    focal_mm = None
    if "FocalLength" in exif:
        focal_mm = _rational_to_float(exif["FocalLength"])

    if focal_mm is not None:
        init_f_px = (focal_mm * 1e-3) / float(pixel_size_m)
    else:
        init_f_px = 4636.912

    return width, height, init_f_px


def d2r(v):
    return v * math.pi / 180.0


def Rx(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float64)


def Ry(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=np.float64)


def Rz(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float64)


def build_R_i2g(omega_deg, phi_deg, kappa_deg):
    return Rx(d2r(float(omega_deg))) @ Ry(d2r(float(phi_deg))) @ Rz(d2r(float(kappa_deg)))


def apply_camera_frame_transform(R_cw_raw, mode):
    if mode == "raw":
        return R_cw_raw.copy()
    if mode == "flip_x":
        return R_cw_raw @ np.diag([-1.0, 1.0, 1.0])
    if mode == "flip_y":
        return R_cw_raw @ np.diag([1.0, -1.0, 1.0])
    if mode == "flip_z":
        return R_cw_raw @ np.diag([1.0, 1.0, -1.0])
    if mode == "flip_xy":
        return R_cw_raw @ np.diag([-1.0, -1.0, 1.0])
    if mode == "flip_xz":
        return R_cw_raw @ np.diag([-1.0, 1.0, -1.0])
    if mode == "flip_yz":
        return R_cw_raw @ np.diag([1.0, -1.0, -1.0])
    raise ValueError(f"Unknown camera frame mode: {mode}")


def robust_stats(v):
    v = np.asarray(v, dtype=np.float64)
    return {
        "n": int(len(v)),
        "mean": float(np.mean(v)),
        "med": float(np.median(v)),
        "rmse": float(np.sqrt(np.mean(v**2))),
        "p95": float(np.percentile(v, 95)),
        "max": float(np.max(v)),
    }


def compute_xy_bbox_from_laz(laz_path: Path, stride: int = 1):
    las = laspy.read(str(laz_path))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)

    stride = max(1, int(stride))
    x = x[::stride]
    y = y[::stride]

    return {
        "xmin": float(np.min(x)),
        "xmax": float(np.max(x)),
        "ymin": float(np.min(y)),
        "ymax": float(np.max(y)),
    }


def filter_tp3d_by_bbox(tp3d, xmin, xmax, ymin, ymax, verbose=1):
    out = {}
    rejected = 0

    for pid, (x, y, z) in tp3d.items():
        if x < xmin or x > xmax or y < ymin or y > ymax:
            rejected += 1
            continue
        out[pid] = (x, y, z)

    if verbose >= 1:
        print(f"[INFO] tp3d filtered by LAZ bbox: kept={len(out)}, rejected={rejected}")

    return out


def distort_brown(x, y, p):
    r2 = x * x + y * y
    r4 = r2 * r2
    radial = 1.0 + p["k1"] * r2 + p["k2"] * r4
    xd = x * radial + 2.0 * p["p1"] * x * y + p["p2"] * (r2 + 2.0 * x * x)
    yd = y * radial + p["p1"] * (r2 + 2.0 * y * y) + 2.0 * p["p2"] * x * y
    return xd, yd


def build_calibration_dataset(eors_df, tp3d, obs_dir, camera_frame_mode, max_points_per_image=100, verbose=1):
    rows = []
    kept_labels = set()
    ignored_images = []

    for row in eors_df.itertuples(index=False):
        label = str(row.label)
        stem = Path(label).stem
        obs_path = Path(obs_dir) / f"{stem}_obs.txt"

        if not obs_path.exists():
            ignored_images.append((stem, "obs file missing"))
            continue

        obs_rows = parse_obs_file(obs_path, max_points_per_image=max_points_per_image)
        if not obs_rows:
            ignored_images.append((stem, "obs file empty"))
            continue

        C = np.array([float(row.X0), float(row.Y0), float(row.Z0)], dtype=np.float64)
        R_i2g = build_R_i2g(row.omega_deg, row.phi_deg, row.kappa_deg)
        R_cw_raw = R_i2g.T
        R_cw = apply_camera_frame_transform(R_cw_raw, camera_frame_mode)
        tvec = -R_cw @ C

        kept_for_image = 0
        rejected_for_image = 0

        for pid, c_obs, l_obs in obs_rows:
            xyz = tp3d.get(pid)
            if xyz is None:
                rejected_for_image += 1
                continue

            xyz = np.asarray(xyz, dtype=np.float64)
            Xc = R_cw @ xyz + tvec
            z = -float(Xc[2])

            if z <= 1e-12:
                rejected_for_image += 1
                continue

            x = float(Xc[0] / z)
            y = float(Xc[1] / z)

            rows.append({
                "label": label,
                "stem": stem,
                "pid": pid,
                "x": x,
                "y": y,
                "c_obs": c_obs,
                "l_obs": l_obs,
            })
            kept_for_image += 1

        if kept_for_image == 0:
            ignored_images.append((stem, f"outside LAZ extent / no valid homologous points (rejected={rejected_for_image})"))
            continue

        kept_labels.add(label)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("Aucune observation exploitable pour la calibration après filtrage par emprise LAZ.")

    return df, kept_labels, ignored_images


def predict_brown(df, p):
    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    xd, yd = distort_brown(x, y, p)
    c_pred = p["fx"] * xd + p["cx"]
    l_pred = p["fy"] * yd + p["cy"]
    return c_pred, l_pred


def evaluate_brown(df, p):
    c_obs = df["c_obs"].to_numpy(dtype=np.float64)
    l_obs = df["l_obs"].to_numpy(dtype=np.float64)
    c_pred, l_pred = predict_brown(df, p)
    dc = c_obs - c_pred
    dl = l_obs - l_pred
    err = np.sqrt(dc * dc + dl * dl)

    stats = {
        "err": robust_stats(err),
        "dc": robust_stats(dc),
        "dl": robust_stats(dl),
        "mean_dc": float(np.mean(dc)),
        "mean_dl": float(np.mean(dl)),
    }
    return stats, c_pred, l_pred, dc, dl, err


def fit_brown_update(df, p):
    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    c_obs = df["c_obs"].to_numpy(dtype=np.float64)
    l_obs = df["l_obs"].to_numpy(dtype=np.float64)

    r2 = x * x + y * y
    r4 = r2 * r2
    radial = 1.0 + p["k1"] * r2 + p["k2"] * r4

    xd = x * radial + 2.0 * p["p1"] * x * y + p["p2"] * (r2 + 2.0 * x * x)
    yd = y * radial + p["p1"] * (r2 + 2.0 * y * y) + 2.0 * p["p2"] * x * y

    c_pred = p["fx"] * xd + p["cx"]
    l_pred = p["fy"] * yd + p["cy"]

    dc = c_obs - c_pred
    dl = l_obs - l_pred

    n = len(df)
    A = np.zeros((2 * n, 8), dtype=np.float64)
    b = np.zeros((2 * n,), dtype=np.float64)

    A[0::2, 0] = xd
    A[0::2, 1] = 0.0
    A[0::2, 2] = 1.0
    A[0::2, 3] = 0.0
    A[0::2, 4] = p["fx"] * x * r2
    A[0::2, 5] = p["fx"] * x * r4
    A[0::2, 6] = p["fx"] * (2.0 * x * y)
    A[0::2, 7] = p["fx"] * (r2 + 2.0 * x * x)
    b[0::2] = dc

    A[1::2, 0] = 0.0
    A[1::2, 1] = yd
    A[1::2, 2] = 0.0
    A[1::2, 3] = 1.0
    A[1::2, 4] = p["fy"] * y * r2
    A[1::2, 5] = p["fy"] * y * r4
    A[1::2, 6] = p["fy"] * (r2 + 2.0 * y * y)
    A[1::2, 7] = p["fy"] * (2.0 * x * y)
    b[1::2] = dl

    sol, *_ = np.linalg.lstsq(A, b, rcond=None)

    return {
        "dfx": float(sol[0]),
        "dfy": float(sol[1]),
        "dcx": float(sol[2]),
        "dcy": float(sol[3]),
        "dk1": float(sol[4]),
        "dk2": float(sol[5]),
        "dp1": float(sol[6]),
        "dp2": float(sol[7]),
    }


def estimate_brown_camera(df, width, height, init_f_px, num_iter=5):
    p = {
        "fx": float(init_f_px),
        "fy": float(init_f_px),
        "cx": float(width) / 2.0,
        "cy": float(height) / 2.0,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    }

    last = None
    for it in range(num_iter + 1):
        stats, c_pred, l_pred, dc, dl, err = evaluate_brown(df, p)

        print(f"\n[CALIB ITER {it}]")
        print(
            f"fx={p['fx']:.9f}, fy={p['fy']:.9f}, "
            f"cx={p['cx']:.9f}, cy={p['cy']:.9f}, "
            f"k1={p['k1']:.12e}, k2={p['k2']:.12e}, "
            f"p1={p['p1']:.12e}, p2={p['p2']:.12e}"
        )
        print("err =", stats["err"])
        print(f"mean_dc = {stats['mean_dc']:.9f}")
        print(f"mean_dl = {stats['mean_dl']:.9f}")

        last = (stats, c_pred, l_pred, dc, dl, err)

        if it == num_iter:
            break

        upd = fit_brown_update(df, p)
        print("[CALIB UPDATE]", upd)

        p["fx"] += upd["dfx"]
        p["fy"] += upd["dfy"]
        p["cx"] += upd["dcx"]
        p["cy"] += upd["dcy"]
        p["k1"] += upd["dk1"]
        p["k2"] += upd["dk2"]
        p["p1"] += upd["dp1"]
        p["p2"] += upd["dp2"]

    return p, last


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


def filter_xy_bbox(xyz: np.ndarray, rgb: np.ndarray | None, xmin=None, xmax=None, ymin=None, ymax=None):
    if xmin is None and xmax is None and ymin is None and ymax is None:
        return xyz, rgb

    mask = np.ones(len(xyz), dtype=bool)

    if xmin is not None:
        mask &= xyz[:, 0] >= float(xmin)
    if xmax is not None:
        mask &= xyz[:, 0] <= float(xmax)
    if ymin is not None:
        mask &= xyz[:, 1] >= float(ymin)
    if ymax is not None:
        mask &= xyz[:, 1] <= float(ymax)

    xyz2 = xyz[mask]
    rgb2 = None if rgb is None else rgb[mask]
    return xyz2, rgb2


def project_point_brown(R_cw, t_cw, intr, xyz, z_positive):
    Xc = R_cw @ xyz + t_cw
    z_raw = float(Xc[2])

    if z_positive:
        if z_raw <= 1e-9:
            return None
        z = z_raw
    else:
        if z_raw >= -1e-9:
            return None
        z = -z_raw

    x = Xc[0] / z
    y = Xc[1] / z
    xd, yd = distort_brown(x, y, intr)
    u = intr["fx"] * xd + intr["cx"]
    v = intr["fy"] * yd + intr["cy"]
    return np.array([u, v], dtype=np.float64), z


def build_frames_from_eors(eors_df, image_index, kept_labels, intr_scaled, camera_frame_mode, verbose=1):
    frames = []

    for row in eors_df.itertuples(index=False):
        stem = Path(str(row.label)).stem
        src_img = image_index.get(stem)
        if src_img is None:
            continue
        if str(row.label) not in kept_labels:
            continue

        center = np.array([float(row.X0), float(row.Y0), float(row.Z0)], dtype=np.float64)
        R_i2g = build_R_i2g(row.omega_deg, row.phi_deg, row.kappa_deg)

        R_cw_raw = R_i2g.T
        R_cw_proj = apply_camera_frame_transform(R_cw_raw, camera_frame_mode)

        # Convention finale unique : caméra vers le bas
        R_cw_final = R_cw_proj @ np.diag([1.0, -1.0, -1.0])
        tvec_final = -R_cw_final @ center

        det_final = float(np.linalg.det(R_cw_final))
        if det_final <= 0:
            log(f"[WARN] Rotation finale invalide pour {stem}, det={det_final}", 1, verbose)
            continue

        rot_final = R.from_matrix(R_cw_final)
        qx, qy, qz, qw = rot_final.as_quat()
        qvec = np.array([qw, qx, qy, qz], dtype=np.float64)

        frames.append({
            "image_id": len(frames) + 1,
            "camera_id": 1,
            "frame_name": None,
            "source_stem": stem,
            "source_image": str(src_img),
            "center": center,
            "qvec": qvec,
            "tvec": tvec_final,
            "R_cw": R_cw_final,
            "width": intr_scaled["width"],
            "height": intr_scaled["height"],
            "fx": intr_scaled["fx"],
            "fy": intr_scaled["fy"],
            "cx": intr_scaled["cx"],
            "cy": intr_scaled["cy"],
        })

    return frames


def build_synthetic_observations(frames, pts_xyz, intr, num_terrain_points=None, verbose=1):
    num_pts_total = len(pts_xyz)

    if num_pts_total == 0:
        return {fr["image_id"]: [] for fr in frames}, []

    if num_terrain_points is None or num_terrain_points <= 0 or num_terrain_points >= num_pts_total:
        selected_indices = np.arange(num_pts_total, dtype=np.int64)
    else:
        selected_indices = np.linspace(
            0,
            num_pts_total - 1,
            num=num_terrain_points,
            dtype=np.int64
        )

    selected_indices = np.unique(selected_indices)

    observations_by_image = {fr["image_id"]: [] for fr in frames}
    tracks_by_point = [[] for _ in range(num_pts_total)]

    log(
        f"  Reprojection de {len(selected_indices)} point(s) terrain sur {num_pts_total} disponible(s) "
        f"vers {len(frames)} image(s)...",
        1,
        verbose,
    )
    log("  Convention finale exportée: caméra vers le bas, z_positive=True", 1, verbose)

    iterable = selected_indices
    use_tqdm = verbose >= 1 and tqdm is not None
    if use_tqdm:
        iterable = tqdm(
            selected_indices,
            total=len(selected_indices),
            desc="Reprojection points terrain",
            unit="pt",
        )

    width = intr["width"]
    height = intr["height"]

    for pt_idx in iterable:
        xyz = pts_xyz[pt_idx]
        point3d_id = int(pt_idx + 1)

        for fr in frames:
            proj = project_point_brown(
                fr["R_cw"],
                fr["tvec"],
                intr,
                xyz,
                z_positive=True,
            )

            if proj is None:
                continue

            uv, _ = proj
            u, v = uv

            if not (0.0 <= u < width and 0.0 <= v < height):
                continue

            point2d_idx = len(observations_by_image[fr["image_id"]])

            observations_by_image[fr["image_id"]].append({
                "xy": np.asarray(uv, dtype=np.float64),
                "point3d_id": point3d_id,
            })

            tracks_by_point[pt_idx].append({
                "image_id": int(fr["image_id"]),
                "point2d_idx": int(point2d_idx),
            })

    for track in tracks_by_point:
        track.sort(key=lambda x: (x["image_id"], x["point2d_idx"]))

    for fr in frames:
        image_id = fr["image_id"]
        nobs = len(observations_by_image[image_id])
        if nobs == 0:
            log(f"[INFO] image ignored for export: {fr['source_stem']} (0 observation after LAZ reprojection)", 2, verbose)
        else:
            log(f"  Image {image_id}: {nobs} observations", 1, verbose)

    tracked_points = sum(1 for tr in tracks_by_point if len(tr) > 0)
    log(f"  Points terrain ayant au moins une observation: {tracked_points}", 1, verbose)

    return observations_by_image, tracks_by_point


def filter_points_with_tracks(pts_xyz, pts_rgb, tracks_by_point, observations_by_image, verbose=1, remap=False):
    kept_old_indices = [i for i, tr in enumerate(tracks_by_point) if len(tr) > 0]

    if len(kept_old_indices) == 0:
        pts_xyz_kept = pts_xyz[:0].copy()
        pts_rgb_kept = None if pts_rgb is None else pts_rgb[:0].copy()
        tracks_kept = []
        observations_new = {k: [] for k in observations_by_image}
        old_to_new_point_id = {}
        point3d_ids_kept = np.zeros((0,), dtype=np.int64)
        return pts_xyz_kept, pts_rgb_kept, tracks_kept, observations_new, old_to_new_point_id, point3d_ids_kept

    point3d_ids_kept = np.asarray([old_idx + 1 for old_idx in kept_old_indices], dtype=np.int64)

    if remap:
        old_to_new_point_id = {old_idx + 1: new_id for new_id, old_idx in enumerate(kept_old_indices, start=1)}
    else:
        old_to_new_point_id = {old_idx + 1: old_idx + 1 for old_idx in kept_old_indices}

    pts_xyz_kept = pts_xyz[kept_old_indices]
    pts_rgb_kept = None if pts_rgb is None else pts_rgb[kept_old_indices]
    tracks_kept = [tracks_by_point[i] for i in kept_old_indices]

    if remap:
        n_old = len(tracks_by_point)
        lut = np.zeros(n_old + 1, dtype=np.int64)
        for new_pid, old_idx in enumerate(kept_old_indices, start=1):
            lut[old_idx + 1] = new_pid

        observations_new = {}
        items = observations_by_image.items()
        if verbose >= 1 and tqdm is not None:
            items = tqdm(items, total=len(observations_by_image), desc="Remap observations", unit="img")

        for image_id, obs_list in items:
            new_obs_list = []
            append = new_obs_list.append

            for obs in obs_list:
                old_pid = obs["point3d_id"]
                if 0 < old_pid <= n_old:
                    new_pid = int(lut[old_pid])
                    if new_pid != 0:
                        append({
                            "xy": obs["xy"],
                            "point3d_id": new_pid,
                        })

            observations_new[image_id] = new_obs_list

        point3d_ids_kept = np.arange(1, len(kept_old_indices) + 1, dtype=np.int64)
    else:
        observations_new = observations_by_image

    return pts_xyz_kept, pts_rgb_kept, tracks_kept, observations_new, old_to_new_point_id, point3d_ids_kept


def filter_frames_with_observations_and_remap(frames, observations_by_image, tracks_by_point, verbose=1):
    kept_frames = []
    old_to_new_image_id = {}

    for fr in frames:
        old_id = fr["image_id"]
        if not observations_by_image.get(old_id):
            continue

        new_id = len(kept_frames) + 1
        new_fr = dict(fr)
        new_fr["image_id"] = new_id
        kept_frames.append(new_fr)
        old_to_new_image_id[old_id] = new_id

    if not kept_frames:
        return [], {}, [[] for _ in range(len(tracks_by_point))], {}

    new_observations_by_image = {}
    for old_id, new_id in old_to_new_image_id.items():
        new_observations_by_image[new_id] = observations_by_image.get(old_id, [])

    max_old_image_id = max(fr["image_id"] for fr in frames) if frames else 0
    image_lut = np.zeros(max_old_image_id + 1, dtype=np.int32)
    for old_id, new_id in old_to_new_image_id.items():
        image_lut[int(old_id)] = int(new_id)

    new_tracks_by_point = [None] * len(tracks_by_point)
    for i, track in enumerate(tracks_by_point):
        if not track:
            new_tracks_by_point[i] = []
            continue

        remapped_track = []
        for tr in track:
            old_image_id = int(tr["image_id"])
            if 0 < old_image_id < len(image_lut):
                new_image_id = int(image_lut[old_image_id])
                if new_image_id != 0:
                    remapped_track.append({
                        "image_id": new_image_id,
                        "point2d_idx": int(tr["point2d_idx"]),
                    })

        if len(remapped_track) > 1:
            remapped_track.sort(key=lambda x: (x["image_id"], x["point2d_idx"]))

        new_tracks_by_point[i] = remapped_track

    log(f"  Images conservées après filtrage par observations: {len(kept_frames)}/{len(frames)}", 1, verbose)
    return kept_frames, new_observations_by_image, new_tracks_by_point, old_to_new_image_id


def scale_camera_model(intr, factor):
    factor = max(float(factor), 1.0)
    width2 = max(1, int(round(intr["width"] / factor)))
    height2 = max(1, int(round(intr["height"] / factor)))

    out = dict(intr)
    out["width"] = width2
    out["height"] = height2
    out["fx"] = intr["fx"] / factor
    out["fy"] = intr["fy"] / factor
    out["cx"] = intr["cx"] / factor
    out["cy"] = intr["cy"] / factor
    return out


def write_cameras_txt_opencv(path: Path, intr):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(
            "1 OPENCV "
            f"{intr['width']} {intr['height']} "
            f"{intr['fx']:.12f} {intr['fy']:.12f} "
            f"{intr['cx']:.12f} {intr['cy']:.12f} "
            f"{intr['k1']:.12e} {intr['k2']:.12e} "
            f"{intr['p1']:.12e} {intr['p2']:.12e}\n"
        )


def write_images_txt(path: Path, frames, observations_by_image):
    with open(path, "w", encoding="utf-8") as f:
        write = f.write

        write("# Image list with two lines of data per image:\n")
        write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        write(f"# Number of images: {len(frames)}\n")

        for fr in frames:
            q = fr["qvec"]
            t = fr["tvec"]
            image_id = fr["image_id"]
            camera_id = fr["camera_id"]

            write(
                f"{image_id} "
                f"{q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f} "
                f"{t[0]:.12f} {t[1]:.12f} {t[2]:.12f} "
                f"{camera_id} {fr['frame_name']}\n"
            )

            obs = observations_by_image.get(image_id, [])
            write(" ".join(
                f"{o['xy'][0]:.6f} {o['xy'][1]:.6f} {o['point3d_id']}"
                for o in obs
            ))
            write("\n")


def write_points3D_txt(path: Path, pts_xyz, pts_rgb=None, tracks_by_point=None, point3d_ids=None, verbose=1):
    if pts_rgb is None:
        pts_rgb = np.full((len(pts_xyz), 3), 200, dtype=np.uint8)

    if tracks_by_point is None:
        tracks_by_point = [[] for _ in range(len(pts_xyz))]

    if point3d_ids is None:
        point3d_ids = np.arange(1, len(pts_xyz) + 1, dtype=np.int64)

    if len(point3d_ids) != len(pts_xyz):
        raise ValueError(
            f"point3d_ids et pts_xyz doivent avoir la même longueur "
            f"(ids={len(point3d_ids)} vs pts={len(pts_xyz)})"
        )

    iterable = zip(point3d_ids, pts_xyz, pts_rgb, tracks_by_point)
    if verbose >= 1 and tqdm is not None:
        iterable = tqdm(iterable, total=len(pts_xyz), desc="Écriture points3D.txt", unit="pt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(pts_xyz)}\n")

        for pid, p, c, track in iterable:
            parts = [
                str(int(pid)),
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


def apply_transform_to_points(xyz: np.ndarray, T4: np.ndarray, scale: float):
    if len(xyz) == 0:
        return xyz.copy()

    xyz_h = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
    xyz_t = (T4 @ xyz_h.T).T[:, :3]
    xyz_t *= float(scale)
    return xyz_t


def build_transforms_json(path: Path, frames, intr, extra_c2w_right_multiply=None, applied_transform=None, applied_scale=None):
    if not frames:
        raise ValueError("Aucune frame pour transforms.json")

    data = {
        "w": intr["width"],
        "h": intr["height"],
        "fl_x": intr["fx"],
        "fl_y": intr["fy"],
        "cx": intr["cx"],
        "cy": intr["cy"],
        "k1": intr["k1"],
        "k2": intr["k2"],
        "p1": intr["p1"],
        "p2": intr["p2"],
        "camera_model": "OPENCV",
        "ply_file_path": "sparse_pc.ply",
        "frames": []
    }

    applied_transform_4x4 = None
    if applied_transform is not None:
        applied_transform = np.asarray(applied_transform, dtype=np.float64)
        if applied_transform.shape == (4, 4):
            applied_transform_4x4 = applied_transform
            data["applied_transform"] = applied_transform[:3, :4].tolist()
        elif applied_transform.shape == (3, 4):
            applied_transform_4x4 = np.eye(4, dtype=np.float64)
            applied_transform_4x4[:3, :4] = applied_transform
            data["applied_transform"] = applied_transform.tolist()
        else:
            raise ValueError(f"applied_transform doit être 3x4 ou 4x4, reçu {applied_transform.shape}")

    if applied_scale is not None:
        data["applied_scale"] = float(applied_scale)

    extra = None
    if extra_c2w_right_multiply is not None:
        extra = np.asarray(extra_c2w_right_multiply, dtype=np.float64)
        if extra.shape != (4, 4):
            raise ValueError("extra_c2w_right_multiply doit être 4x4")

    for fr in frames:
        R_cw = np.asarray(fr["R_cw"], dtype=np.float64)
        t_cw = np.asarray(fr["tvec"], dtype=np.float64).reshape(3, 1)

        w2c = np.concatenate([R_cw, t_cw], axis=1)
        w2c = np.concatenate([w2c, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)], axis=0)

        c2w = np.linalg.inv(w2c)

        # Convention Nerfstudio depuis COLMAP
        c2w[0:3, 1:3] *= -1

        # Correction visuelle supplémentaire uniquement viewer
        if extra is not None:
            c2w = c2w @ extra

        if applied_transform_4x4 is not None:
            c2w = applied_transform_4x4 @ c2w

        if applied_scale is not None:
            c2w = c2w.copy()
            c2w[:3, 3] *= float(applied_scale)

        data["frames"].append({
            "file_path": f'./images/{fr["frame_name"]}',
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": fr["image_id"],
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_scene_normalization_json(path: Path, T4: np.ndarray, scale: float, centers: np.ndarray):
    centroid = centers.mean(axis=0) if len(centers) > 0 else np.zeros(3, dtype=np.float64)

    data = {
        "origin_world": centroid.tolist(),
        "scale": float(scale),
        "translation_matrix_4x4": T4.tolist(),
        "formula_points": "X_normalized = scale * (T @ [X_world, 1])[:3]",
        "formula_camera_centers": "C_normalized = scale * (T @ [C_world, 1])[:3]",
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def parse_images_txt_for_validation(path: Path):
    images = {}

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line1 = lines[i].strip()

        if not line1 or line1.startswith("#"):
            i += 1
            continue

        parts = line1.split()
        if len(parts) < 10:
            i += 1
            continue

        image_id = int(parts[0])
        name = parts[9]

        obs = []
        line2 = ""
        if i + 1 < len(lines):
            line2 = lines[i + 1].strip()

        if line2 and not line2.startswith("#"):
            vals = line2.split()
            if len(vals) % 3 != 0:
                raise ValueError(
                    f"images.txt invalide: image_id={image_id}, "
                    f"la ligne POINTS2D ne contient pas un multiple de 3 valeurs"
                )

            for point2d_idx in range(len(vals) // 3):
                x = float(vals[3 * point2d_idx + 0])
                y = float(vals[3 * point2d_idx + 1])
                point3d_id = int(vals[3 * point2d_idx + 2])

                obs.append({
                    "xy": (x, y),
                    "point3d_id": point3d_id,
                    "point2d_idx": point2d_idx,
                })

        images[image_id] = {
            "name": name,
            "observations": obs,
        }

        i += 2

    return images


def parse_points3d_txt_for_validation(path: Path):
    points = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 8:
                raise ValueError(f"points3D.txt invalide ligne {line_no}: moins de 8 champs")

            point3d_id = int(parts[0])

            track_parts = parts[8:]
            if len(track_parts) % 2 != 0:
                raise ValueError(
                    f"points3D.txt invalide ligne {line_no}: TRACK[] n'a pas un nombre pair d'éléments"
                )

            track = []
            for i in range(0, len(track_parts), 2):
                image_id = int(track_parts[i])
                point2d_idx = int(track_parts[i + 1])
                track.append((image_id, point2d_idx))

            points[point3d_id] = {
                "track": track
            }

    return points


def validate_colmap_text_model(images_txt: Path, points3d_txt: Path, verbose: int = 1):
    images = parse_images_txt_for_validation(images_txt)
    points = parse_points3d_txt_for_validation(points3d_txt)

    errors = []
    warnings = []

    image_obs_lookup = {}
    for image_id, im in images.items():
        image_obs_lookup[image_id] = im["observations"]

    for point3d_id, pdata in points.items():
        track = pdata["track"]

        for image_id, point2d_idx in track:
            if image_id not in images:
                errors.append(f"Point3D {point3d_id}: image_id {image_id} absent de images.txt")
                continue

            obs = image_obs_lookup[image_id]

            if point2d_idx < 0 or point2d_idx >= len(obs):
                errors.append(
                    f"Point3D {point3d_id}: point2d_idx {point2d_idx} invalide "
                    f"pour image_id {image_id} (taille={len(obs)})"
                )
                continue

            linked_pid = obs[point2d_idx]["point3d_id"]
            if linked_pid != point3d_id:
                errors.append(
                    f"Incohérence track: Point3D {point3d_id} -> "
                    f"(image {image_id}, idx {point2d_idx}) "
                    f"mais images.txt référence {linked_pid}"
                )

    for image_id, im in images.items():
        for o in im["observations"]:
            pid = o["point3d_id"]
            idx = o["point2d_idx"]

            if pid == -1:
                continue

            if pid not in points:
                errors.append(
                    f"Observation orpheline: image {image_id}, idx {idx} -> "
                    f"point3D_id {pid} absent de points3D.txt"
                )
                continue

            track = points[pid]["track"]
            if (image_id, idx) not in track:
                errors.append(
                    f"Incohérence inverse: image {image_id}, idx {idx} -> point {pid}, "
                    f"mais le track du point ne contient pas cette paire"
                )

    points_without_track = [pid for pid, pdata in points.items() if len(pdata["track"]) == 0]
    if points_without_track:
        warnings.append(f"{len(points_without_track)} point(s) 3D sans track")

    image_obs_count = sum(len(im["observations"]) for im in images.values())
    tracked_image_obs_count = sum(
        1 for im in images.values() for o in im["observations"] if o["point3d_id"] != -1
    )

    if verbose >= 1:
        log(
            f"[VALIDATION] images={len(images)}, points3D={len(points)}, "
            f"obs_total={image_obs_count}, obs_trackees={tracked_image_obs_count}",
            1,
            verbose,
        )

    for w in warnings:
        log(f"[VALIDATION][WARN] {w}", 1, verbose)

    if errors:
        for e in errors[:50]:
            log(f"[VALIDATION][ERROR] {e}", 1, verbose)

        if len(errors) > 50:
            log(f"[VALIDATION][ERROR] ... {len(errors) - 50} erreur(s) supplémentaire(s)", 1, verbose)

        return False, errors, warnings

    log("[VALIDATION] Modèle texte COLMAP cohérent.", 1, verbose)
    return True, errors, warnings


def try_write_colmap_bin(sparse_dir: Path, verbose: int):
    images_txt = sparse_dir / "images.txt"
    points3d_txt = sparse_dir / "points3D.txt"
    cameras_txt = sparse_dir / "cameras.txt"

    if not images_txt.exists() or not points3d_txt.exists() or not cameras_txt.exists():
        log("[WARN] Fichiers texte COLMAP incomplets: conversion binaire ignorée.", 2, verbose)
        return False

    ok, errors, warnings = validate_colmap_text_model(
        images_txt=images_txt,
        points3d_txt=points3d_txt,
        verbose=verbose,
    )

    if not ok:
        log("[WARN] Le modèle texte est incohérent: conversion .bin annulée.", 1, verbose)
        return False

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
        log(f"[WARN] Impossible de générer les .bin avec pycolmap: {e}", 1, verbose)
        return False
        
def get_visual_downward_transform_4x4():
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    return T

def get_nerfstudio_axis_transform_4x4():
    T = np.eye(4, dtype=np.float64)
    T[:3, :4] = np.array([
        [1.0,  0.0,  0.0, 0.0],
        [0.0,  0.0,  1.0, 0.0],
        [0.0, -1.0,  0.0, 0.0],
    ], dtype=np.float64)
    return T
        
def write_ply_xyzrgb(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None):
    n = len(xyz)

    if rgb is None:
        rgb = np.full((n, 3), 200, dtype=np.uint8)
    else:
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    xyz_out = np.asarray(xyz, dtype=np.float64)

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

        for p, c in zip(xyz_out, rgb):
            f.write(
                f"{float(p[0]):.12f} {float(p[1]):.12f} {float(p[2]):.12f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )

def prepare_selected_images(source_paths, output_dir: Path, factor: float, jpeg_quality: int,
                            verbose: int, image_convertor_script: Path):
    ensure_dir(output_dir)

    if not image_convertor_script.exists():
        raise FileNotFoundError(f"Script de conversion introuvable: {image_convertor_script}")

    produced = {}

    for src_path in source_paths:
        src_path = Path(src_path)
        stem = src_path.stem
        ext = src_path.suffix.lower()

        if ext in {".jp2", ".tif", ".tiff"}:
            out_path = output_dir / f"{stem}.jpg"
            output_mode = "--jpg"

        elif ext in {".jpg", ".jpeg", ".png", ".bmp"}:
            if factor <= 1.0:
                out_path = output_dir / src_path.name
                log(f"  Copie: {src_path.name}", 2, verbose)
                shutil.copy2(src_path, out_path)
                produced[stem] = out_path
                continue

            out_path = output_dir / src_path.name
            if ext in {".jpg", ".jpeg"}:
                output_mode = "--jpg"
            elif ext == ".png":
                output_mode = "--png"
            elif ext == ".bmp":
                output_mode = "--png"
                out_path = output_dir / f"{stem}.png"
            else:
                output_mode = "--jpg"
                out_path = output_dir / f"{stem}.jpg"
        else:
            log(f"[WARN] Format non géré ignoré: {src_path.name}", 2, verbose)
            continue

        cmd = [
            sys.executable,
            str(image_convertor_script),
            "--input", str(src_path),
            "--output", str(out_path),
            "--factor", str(factor),
            output_mode,
            "--jpeg-quality", str(jpeg_quality),
        ]

        if verbose >= 1:
            cmd.append("--verbose")

        log(f"  Conversion: {src_path.name} -> {out_path.name}", 2, verbose)
        subprocess.run(cmd, check=True)

        if not out_path.exists():
            raise FileNotFoundError(f"Image convertie introuvable après conversion: {out_path}")

        produced[stem] = out_path

    return produced

def main():
    ap = argparse.ArgumentParser(
        description="Génère un modèle COLMAP cohérent depuis eors+tp3d+obs+laz, et une sortie viewer séparée avec caméras orientées vers le bas."
    )
    ap.add_argument("--eors", required=True, help="Fichier eors.txt")
    ap.add_argument("--tp3d", required=True, help="Fichier tp.txt")
    ap.add_argument("--laz", required=True, help="Fichier .LAZ")
    ap.add_argument("--images", required=True, help="Dossier des images source")
    ap.add_argument("--out", required=True, help="Dossier de sortie")
    ap.add_argument("--subsample", type=int, default=10, help="Facteur de sous-échantillonnage initial du LAZ")
    ap.add_argument("--image-factor", type=float, default=1.0,
                    help="Facteur de sous-échantillonnage des images (2 = largeur/hauteur divisées par 2)")
    ap.add_argument("--jpeg-quality", type=int, default=95, help="Qualité JPEG de sortie")
    ap.add_argument("--num-terrain-points", type=int, default=5000,
                    help="Nombre de points terrain à reprojeter dans toutes les images")
    ap.add_argument("--xmin", type=float, default=None)
    ap.add_argument("--xmax", type=float, default=None)
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--max-points-per-image", type=int, default=100)
    ap.add_argument("--num-calib-iters", type=int, default=5)
    ap.add_argument("--pixel-size", type=float, default=4.52e-6)
    ap.add_argument("--camera-frame", type=str, default="raw",
                    choices=["raw", "flip_x", "flip_y", "flip_z", "flip_xy", "flip_xz", "flip_yz"])
    ap.add_argument("--remap", action="store_true")
    ap.add_argument("--z-negative", action="store_true",
                    help="Conserve la convention photogrammétrique valide: z<0 visible.")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    args = ap.parse_args()

    t0 = time.perf_counter()
    dt_start = datetime.now()
    log(f"[TIME] Début: {dt_start.strftime('%Y-%m-%d %H:%M:%S')}", 1, args.verbose)

    eors_path = Path(args.eors)
    tp3d_path = Path(args.tp3d)
    laz_path = Path(args.laz)
    images_dir = Path(args.images)
    out_dir = Path(args.out)

    script_dir = Path(__file__).resolve().parent
    image_convertor_script = script_dir / "image_convertor.py"

    out_images = out_dir / "images"
    out_colmap = out_dir / "colmap"
    out_sparse = out_colmap / "sparse" / "0"
    out_models_0 = out_sparse / "models" / "0"

    sparse_pc_ply = out_dir / "sparse_pc.ply"
    normalization_json = out_dir / "scene_normalization.json"
    intrinsics_json = out_dir / "estimated_brown_camera.json"

    transforms_colmap_like = out_dir / "transforms_colmap_like.json"
    transforms_view_down = out_dir / "transforms_view_down.json"

    z_positive = not args.z_negative
    log(f"[CONFIG] z_positive={z_positive}", 1, args.verbose)
    log(f"[CONFIG] camera_frame={args.camera_frame}", 1, args.verbose)

    for d in [out_images, out_sparse, out_models_0]:
        ensure_dir(d)

    log("[1/9] Indexation des images existantes...", 1, args.verbose)
    image_index = build_image_index(images_dir)
    log(f"  {len(image_index)} images indexées.", 1, args.verbose)

    if not image_index:
        print("Aucune image compatible trouvée dans le dossier fourni.")
        sys.exit(2)

    log("[2/9] Lecture eors + tp3d...", 1, args.verbose)
    eors_df = load_eors(eors_path)
    tp3d = load_tp3d(tp3d_path)
    log(f"  {len(eors_df)} orientations chargées", 1, args.verbose)
    log(f"  {len(tp3d)} points 3D chargés", 1, args.verbose)

    log("[2b/9] Calcul de l'emprise XY du LAZ...", 1, args.verbose)
    laz_bbox = compute_xy_bbox_from_laz(laz_path, stride=max(1, args.subsample))

    if args.xmin is not None:
        laz_bbox["xmin"] = max(laz_bbox["xmin"], float(args.xmin))
    if args.xmax is not None:
        laz_bbox["xmax"] = min(laz_bbox["xmax"], float(args.xmax))
    if args.ymin is not None:
        laz_bbox["ymin"] = max(laz_bbox["ymin"], float(args.ymin))
    if args.ymax is not None:
        laz_bbox["ymax"] = min(laz_bbox["ymax"], float(args.ymax))

    log(
        f"  LAZ bbox used for tp3d filtering: "
        f"x=[{laz_bbox['xmin']:.3f}, {laz_bbox['xmax']:.3f}] "
        f"y=[{laz_bbox['ymin']:.3f}, {laz_bbox['ymax']:.3f}]",
        1,
        args.verbose,
    )

    tp3d = filter_tp3d_by_bbox(
        tp3d,
        xmin=laz_bbox["xmin"],
        xmax=laz_bbox["xmax"],
        ymin=laz_bbox["ymin"],
        ymax=laz_bbox["ymax"],
        verbose=args.verbose,
    )

    if len(tp3d) == 0:
        print("Aucun point tp3d dans l'emprise du LAZ.")
        sys.exit(10)

    width, height, init_f_px = infer_image_size_and_initial_f(images_dir, pixel_size_m=args.pixel_size)
    log(f"  Taille image inférée: {width}x{height}", 1, args.verbose)
    log(f"  Focale initiale: {init_f_px:.6f} px", 1, args.verbose)

    log("[3/9] Construction du dataset de calibration...", 1, args.verbose)
    calib_df, kept_labels, ignored_images = build_calibration_dataset(
        eors_df=eors_df,
        tp3d=tp3d,
        obs_dir=images_dir,
        camera_frame_mode=args.camera_frame,
        max_points_per_image=args.max_points_per_image,
        verbose=args.verbose,
    )
    log(f"  {len(calib_df)} observations utilisables", 1, args.verbose)
    log(f"  {len(kept_labels)} image(s) conservée(s) pour la calibration", 1, args.verbose)

    log("[4/9] Estimation de la caméra Brown...", 1, args.verbose)
    intr, calib_last = estimate_brown_camera(
        df=calib_df,
        width=width,
        height=height,
        init_f_px=init_f_px,
        num_iter=args.num_calib_iters,
    )
    calib_stats, _, _, _, _, _ = calib_last
    intr["width"] = width
    intr["height"] = height

    print("\n[FINAL BROWN CAMERA]")
    print(intr)
    print("[FINAL CALIB RESIDUALS]")
    print(calib_stats["err"])

    with open(intrinsics_json, "w", encoding="utf-8") as f:
        json.dump({
            "width": intr["width"],
            "height": intr["height"],
            "fx": intr["fx"],
            "fy": intr["fy"],
            "cx": intr["cx"],
            "cy": intr["cy"],
            "k1": intr["k1"],
            "k2": intr["k2"],
            "p1": intr["p1"],
            "p2": intr["p2"],
            "residuals": calib_stats["err"],
            "colmap_projection_convention": {
                "camera_frame": args.camera_frame,
                "z_positive": False,
                "formula": "standard",
            }
        }, f, indent=2)

    log("[5/9] Construction des poses depuis eors...", 1, args.verbose)
    factor = max(float(args.image_factor), 1.0)
    intr_scaled = scale_camera_model(intr, factor)

    frames = build_frames_from_eors(
        eors_df=eors_df,
        image_index=image_index,
        kept_labels=kept_labels,
        intr_scaled=intr_scaled,
        camera_frame_mode=args.camera_frame,
        verbose=args.verbose,
    )

    if not frames:
        print("Aucune image commune exploitable entre eors, images et emprise LAZ.")
        sys.exit(3)

    log(f"  {len(frames)} poses retenues", 1, args.verbose)

    log("[6/9] Lecture et sous-échantillonnage du LAZ...", 1, args.verbose)
    pts_xyz_raw, pts_rgb = read_laz_points(laz_path, args.subsample)
    log(f"  {len(pts_xyz_raw)} points conservés après sous-échantillonnage initial.", 1, args.verbose)

    bbox_enabled = any(v is not None for v in (args.xmin, args.xmax, args.ymin, args.ymax))
    if bbox_enabled:
        log("[6b/9] Filtrage du LAZ par bbox XY...", 1, args.verbose)
        before_bbox = len(pts_xyz_raw)
        pts_xyz_raw, pts_rgb = filter_xy_bbox(
            pts_xyz_raw, pts_rgb,
            xmin=args.xmin, xmax=args.xmax,
            ymin=args.ymin, ymax=args.ymax,
        )
        log(f"  {len(pts_xyz_raw)}/{before_bbox} points conservés dans la bbox.", 1, args.verbose)

    if len(pts_xyz_raw) == 0:
        print("Aucun point LAZ conservé après filtrage.")
        sys.exit(4)

    log("[7/9] Génération des observations synthétiques COLMAP...", 1, args.verbose)
    observations_by_image, tracks_by_point = build_synthetic_observations(
        frames=frames,
        pts_xyz=pts_xyz_raw,
        intr=intr_scaled,
        num_terrain_points=args.num_terrain_points,
        verbose=args.verbose,
    )

    pts_xyz_kept, pts_rgb_kept, tracks_kept, observations_by_image, _, point3d_ids_kept = filter_points_with_tracks(
        pts_xyz=pts_xyz_raw,
        pts_rgb=pts_rgb,
        tracks_by_point=tracks_by_point,
        observations_by_image=observations_by_image,
        verbose=args.verbose,
        remap=args.remap,
    )

    log(f"  Points 3D exportés avec tracks: {len(pts_xyz_kept)}", 1, args.verbose)

    if len(pts_xyz_kept) == 0:
        print("Aucun point 3D avec track après filtrage.")
        sys.exit(5)

    log("[8/9] Filtrage des images sans homologue conservé + préparation images...", 1, args.verbose)
    frames, observations_by_image, tracks_kept, old_to_new_image_id = filter_frames_with_observations_and_remap(
        frames=frames,
        observations_by_image=observations_by_image,
        tracks_by_point=tracks_kept,
        verbose=args.verbose,
    )

    if not frames:
        print("Aucune image ne possède de point homologue conservé.")
        sys.exit(6)

    # Appliquer le même filtrage / remap à frames
    view_by_old_id = {}
    for idx, fr in enumerate(frames, start=1):
        view_by_old_id[idx] = fr

    frames_kept = []
    for old_id, new_id in old_to_new_image_id.items():
        fr = dict(view_by_old_id[old_id])
        fr["image_id"] = new_id
        frames_kept.append(fr)

    frames = frames_kept

    selected_source_paths = [Path(fr["source_image"]) for fr in frames]
    exported_index = prepare_selected_images(
        source_paths=selected_source_paths,
        output_dir=out_images,
        factor=factor,
        jpeg_quality=args.jpeg_quality,
        verbose=args.verbose,
        image_convertor_script=image_convertor_script,
    )

    for fr in frames:
        stem = fr["source_stem"]
        exported_img = exported_index.get(stem)
        if exported_img is None:
            raise RuntimeError(f"Image exportée absente après préparation: {stem}")
        fr["frame_name"] = exported_img.name

    for fr in frames:
        stem = fr["source_stem"]
        exported_img = exported_index.get(stem)
        if exported_img is None:
            raise RuntimeError(f"Image exportée absente après préparation: {stem}")
        fr["frame_name"] = exported_img.name

    log(f"  {len(exported_index)} image(s) utile(s) préparée(s).", 1, args.verbose)

    log("[9/9] Écriture COLMAP + sorties viewer...", 1, args.verbose)
    centers = np.stack([fr["center"] for fr in frames], axis=0)

    applied_transform = get_nerfstudio_axis_transform_4x4()
    applied_scale = 1.0
    visual_down = get_visual_downward_transform_4x4()

    pts_xyz_kept_ns = apply_transform_to_points(pts_xyz_kept, applied_transform, applied_scale)

    write_scene_normalization_json(
        normalization_json,
        applied_transform,
        applied_scale,
        centers,
    )

    write_cameras_txt_opencv(out_sparse / "cameras.txt", intr_scaled)
    write_images_txt(out_sparse / "images.txt", frames, observations_by_image)
    write_points3D_txt(
        out_sparse / "points3D.txt",
        pts_xyz_kept,
        pts_rgb_kept,
        tracks_kept,
        point3d_ids=point3d_ids_kept,
        verbose=args.verbose,
    )
    write_ply_xyzrgb(sparse_pc_ply, pts_xyz_kept_ns, pts_rgb_kept)

    build_transforms_json(
        out_dir / "transforms.json",
        frames,
        intr_scaled,
        extra_c2w_right_multiply=None,
        applied_transform=applied_transform,
        applied_scale=applied_scale,
    )

    bin_ok = try_write_colmap_bin(out_sparse, args.verbose)

    if bin_ok:
        for txt_name in ("cameras.txt", "images.txt", "points3D.txt"):
            txt_path = out_sparse / txt_name
            try:
                txt_path.unlink()
                log(f"[INFO] Supprimé après conversion binaire: {txt_path}", 1, args.verbose)
            except FileNotFoundError:
                pass
            except Exception as e:
                log(f"[WARN] Impossible de supprimer {txt_path}: {e}", 1, args.verbose)

    print("\nTerminé.")
    print(f"Sortie: {out_dir}")
    print(f"Images utiles: {out_images}")
    print(f"COLMAP sparse: {out_sparse}")
    print(f"Sparse PLY viewer: {sparse_pc_ply}")
    print(f"Transforms viewer: {out_dir / 'transforms.json'}")
    print(f"Normalisation: {normalization_json}")
    print(f"Intrinsics estimées: {intrinsics_json}")

    dt_end = datetime.now()
    elapsed_s = time.perf_counter() - t0
    h = int(elapsed_s // 3600)
    m = int((elapsed_s % 3600) // 60)
    s = elapsed_s % 60

    print(f"Heure début: {dt_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Heure fin  : {dt_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Durée totale: {h:02d}:{m:02d}:{int(s):02d}")


if __name__ == "__main__":
    main()

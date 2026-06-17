#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np


def info(msg):
    print(msg)

def frame_key(frame):
    return Path(frame["file_path"]).name
    
def load_transforms(dataset_dir: Path):
    tf_path = dataset_dir / "transforms.json"
    with open(tf_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ply_xyz(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    end_idx = None
    n_vertices = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("element vertex"):
            n_vertices = int(s.split()[-1])
        if s == "end_header":
            end_idx = i
            break

    if end_idx is None or n_vertices is None:
        raise ValueError(f"PLY invalide: {path}")

    pts = []
    for line in lines[end_idx + 1:end_idx + 1 + n_vertices]:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        pts.append([float(parts[0]), float(parts[1]), float(parts[2])])

    return np.asarray(pts, dtype=np.float64)


def get_frame_map(tf):
    return {fr["file_path"]: fr for fr in tf["frames"]}


def extract_camera_centers_and_rotations(tf):
    out = {}
    for fr in tf["frames"]:
        key = Path(fr["file_path"]).name
        T = np.asarray(fr["transform_matrix"], dtype=np.float64)
        R = T[:3, :3]
        C = T[:3, 3]
        out[key] = {
            "file_path": fr["file_path"],
            "center": C,
            "R": R,
        }
    return out


def umeyama_alignment(X, Y, with_scale=True):
    """
    Estimate similarity transform mapping X -> Y
    Y ~= s * R * X + t
    X, Y shape (N, 3)
    """
    assert X.shape == Y.shape
    n, dim = X.shape

    mx = X.mean(axis=0)
    my = Y.mean(axis=0)

    Xc = X - mx
    Yc = Y - my

    cov = (Yc.T @ Xc) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt

    if with_scale:
        var_x = np.mean(np.sum(Xc**2, axis=1))
        scale = np.sum(D * np.diag(S)) / var_x
    else:
        scale = 1.0

    t = my - scale * (R @ mx)
    return scale, R, t


def apply_similarity(X, s, R, t):
    return (s * (R @ X.T)).T + t[None, :]


def rms(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def angle_deg_between(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    v2 = v2 / (np.linalg.norm(v2) + 1e-12)
    c = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def bbox_stats(name, pts):
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    ctr = pts.mean(axis=0)
    rad = np.max(np.linalg.norm(pts - ctr[None, :], axis=1))
    info(f"{name}:")
    info(f"  count   = {len(pts)}")
    info(f"  min     = {mn}")
    info(f"  max     = {mx}")
    info(f"  center  = {ctr}")
    info(f"  radius  = {rad:.6f}")


def project_points(T_wc, pts_world, fl_x, fl_y, cx, cy, w, h):
    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3, 3]

    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc

    Xc = (R_cw @ pts_world.T).T + t_cw[None, :]
    z = Xc[:, 2]

    front = z > 1e-9
    uv = np.zeros((len(pts_world), 2), dtype=np.float64)

    valid = front
    uv[valid, 0] = fl_x * (Xc[valid, 0] / z[valid]) + cx
    uv[valid, 1] = fl_y * (Xc[valid, 1] / z[valid]) + cy

    inside = (
        front &
        (uv[:, 0] >= 0.0) & (uv[:, 0] < w) &
        (uv[:, 1] >= 0.0) & (uv[:, 1] < h)
    )

    return front, inside


def coverage_stats(name, tf, pts):
    w = tf["w"]
    h = tf["h"]
    fl_x = tf["fl_x"]
    fl_y = tf["fl_y"]
    cx = tf["cx"]
    cy = tf["cy"]

    visible_counts = np.zeros(len(pts), dtype=np.int32)
    per_cam_inside = []

    for fr in tf["frames"]:
        T = np.asarray(fr["transform_matrix"], dtype=np.float64)
        front, inside = project_points(T, pts, fl_x, fl_y, cx, cy, w, h)
        visible_counts += inside.astype(np.int32)
        per_cam_inside.append(int(np.count_nonzero(inside)))

    per_cam_inside = np.asarray(per_cam_inside)
    info(f"{name} coverage:")
    info(f"  visible/cam min={per_cam_inside.min()} mean={per_cam_inside.mean():.1f} max={per_cam_inside.max()}")
    info(f"  points seen >=1 view: {np.count_nonzero(visible_counts >= 1)} / {len(pts)}")
    info(f"  points seen >=2 views: {np.count_nonzero(visible_counts >= 2)} / {len(pts)}")
    info(f"  mean views/point: {visible_counts.mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Dataset de référence (Nerfstudio/COLMAP fonctionnel)")
    ap.add_argument("--test", required=True, help="Dataset converti à tester")
    args = ap.parse_args()

    ref_dir = Path(args.ref)
    test_dir = Path(args.test)

    ref_tf = load_transforms(ref_dir)
    test_tf = load_transforms(test_dir)

    ref_ply = load_ply_xyz(ref_dir / ref_tf.get("ply_file_path", "sparse_pc.ply"))
    test_ply = load_ply_xyz(test_dir / test_tf.get("ply_file_path", "sparse_pc.ply"))

    info("=== INTRINSICS ===")
    for k in ["w", "h", "fl_x", "fl_y", "cx", "cy", "camera_model"]:
        info(f"{k}: ref={ref_tf.get(k)} | test={test_tf.get(k)}")

    ref_map = extract_camera_centers_and_rotations(ref_tf)
    test_map = extract_camera_centers_and_rotations(test_tf)

    common = sorted(set(ref_map.keys()) & set(test_map.keys()))
    info(f"\n=== FRAMES ===")
    info(f"ref frames  = {len(ref_map)}")
    info(f"test frames = {len(test_map)}")
    info(f"common      = {len(common)}")

    if len(common) < 3:
        raise RuntimeError("Pas assez de frames communes pour estimer une similitude.")

    X = np.stack([test_map[k]["center"] for k in common], axis=0)
    Y = np.stack([ref_map[k]["center"] for k in common], axis=0)

    before = rms(X, Y)
    s, R_align, t = umeyama_alignment(X, Y, with_scale=True)
    X_aligned = apply_similarity(X, s, R_align, t)
    after = rms(X_aligned, Y)

    info(f"\n=== ALIGNEMENT CAMERAS (test -> ref) ===")
    info(f"RMS avant  = {before:.6f}")
    info(f"RMS après  = {after:.6f}")
    info(f"scale      = {s:.12f}")
    info(f"det(R)     = {np.linalg.det(R_align):.12f}")
    info(f"R =\n{R_align}")
    info(f"t = {t}")

    info(f"\n=== ORIENTATIONS ===")
    # Compare forward axis = camera local z-axis in world coordinates
    # columns of R_wc
    angs = []
    for k in common:
        R_ref = ref_map[k]["R"]
        R_test = test_map[k]["R"]

        # rotate test orientation into ref frame
        R_test_to_ref = R_align @ R_test

        f_ref = R_ref[:, 2]
        f_test = R_test_to_ref[:, 2]
        angs.append(angle_deg_between(f_ref, f_test))

    angs = np.asarray(angs)
    info(f"forward-axis angle error: min={angs.min():.4f} mean={angs.mean():.4f} max={angs.max():.4f} deg")

    info(f"\n=== POINT CLOUD STATS ===")
    bbox_stats("ref ply", ref_ply)
    bbox_stats("test ply", test_ply)

    test_ply_aligned = apply_similarity(test_ply, s, R_align, t)
    bbox_stats("test ply aligned", test_ply_aligned)

    info(f"\n=== COVERAGE ===")
    coverage_stats("ref", ref_tf, ref_ply)
    coverage_stats("test", test_tf, test_ply)

    info(f"\n=== DONE ===")


if __name__ == "__main__":
    main()

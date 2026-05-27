import numpy as np
import argparse
import os
import struct
import json
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement


# ======================
# LOG
# ======================
def log(msg):
    print(f"[INFO] {msg}")


def warn(msg):
    print(f"[WARN] {msg}")


# ======================
# SAFE READ
# ======================
def safe_read(f, size):
    data = f.read(size)
    if len(data) != size:
        raise EOFError(f"Expected {size} bytes, got {len(data)}")
    return data


# ======================
# TRANSFORM
# ======================
def apply_transform(pts, transform, scale=1.0):
    R = np.array(transform[:3, :3], dtype=np.float64)
    t = np.array(transform[:3, 3], dtype=np.float64)
    pts = (pts @ R.T) + t
    return pts * scale


# ======================
# COLMAP LOADER
# ======================
def load_colmap_points(path, transform, scale, max_points=4_000_000):
    pts = []

    with open(path, "rb") as f:
        n = struct.unpack("<Q", safe_read(f, 8))[0]
        log(f"Raw COLMAP points: {n}")

        for i in range(n):
            try:
                safe_read(f, 8)
                x, y, z = struct.unpack("<ddd", safe_read(f, 24))
                safe_read(f, 3)
                safe_read(f, 8)

                track_len = struct.unpack("<Q", safe_read(f, 8))[0]
                f.seek(track_len * 8, 1)

                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    pts.append([x, y, z])

                if len(pts) >= max_points:
                    break

            except Exception as e:
                warn(f"Skipping point {i}: {e}")
                break

    pts = np.array(pts, dtype=np.float32)

    if len(pts) == 0:
        raise ValueError("No valid COLMAP points")

    pts = apply_transform(pts, transform, scale)

    log(f"Valid COLMAP points: {len(pts)}")
    return pts


# ======================
# OUTLIER FILTER (percentile)
# ======================
def filter_outliers(pts, percentile=97):
    center = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - center, axis=1)

    thresh = np.percentile(dist, percentile)
    mask = dist <= thresh

    filtered = pts[mask]

    log(f"Outlier filtering: kept {len(filtered)}/{len(pts)} (p={percentile})")

    return filtered, mask


# ======================
# RADIUS
# ======================
def estimate_radius(pts, factor=3.0):
    n = min(5000, len(pts))
    sample = pts[np.random.choice(len(pts), n, replace=False)]
    tree = cKDTree(sample)
    d, _ = tree.query(sample, k=2)
    return float(np.median(d[:, 1]) * factor)


# ======================
# LOAD PLY
# ======================
def load_ply(path):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
    return xyz, ply


# ======================
# FILTER GAUSSIANS
# ======================
def filter_gaussians(gaussians, colmap_pts, dist):
    tree = cKDTree(colmap_pts)

    valid = np.isfinite(gaussians).all(axis=1)
    d, _ = tree.query(gaussians[valid], k=1)

    mask = np.zeros(len(gaussians), dtype=bool)
    idx = np.where(valid)[0]

    mask[idx] = d < dist
    return mask


# ======================
# WRITE
# ======================
def write_ply(template_ply, mask, out_path):
    vertex = template_ply["vertex"].data
    new_vertex = vertex[mask]
    PlyData([PlyElement.describe(new_vertex, "vertex")], text=False).write(out_path)


# ======================
# MAIN
# ======================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in-ply", required=True)
    parser.add_argument("--points", required=True)
    parser.add_argument("--transform", required=True)
    parser.add_argument("--dist", type=float, default=None)
    parser.add_argument("--out-ply", required=True)
    parser.add_argument("--outlier-percent", type=float, default=97)

    args = parser.parse_args()

    # ======================
    # LOAD TRANSFORM
    # ======================
    with open(args.transform, "r") as f:
        data = json.load(f)

    transform = np.array(data["transform"], dtype=np.float64)
    scale = float(data.get("scale", 1.0))

    # ======================
    # COLMAP
    # ======================
    colmap_pts = load_colmap_points(args.points, transform, scale)

    # OUTLIER CLEANING
    colmap_filtered, _ = filter_outliers(colmap_pts, args.outlier_percent)

    # BBOX of filtered points
    bbox_min = colmap_filtered.min(axis=0)
    bbox_max = colmap_filtered.max(axis=0)

    log(f"BBOX filtered COLMAP: {bbox_min} → {bbox_max}")

    # ======================
    # PLY
    # ======================
    gaussians, ply = load_ply(args.in_ply)

    # ======================
    # RADIUS
    # ======================
    dist = args.dist or estimate_radius(colmap_filtered)
    log(f"Radius: {dist:.4f}")

    # ======================
    # FILTER DISTANCE
    # ======================
    tree = cKDTree(colmap_filtered)
    valid = np.isfinite(gaussians).all(axis=1)

    d, _ = tree.query(gaussians[valid], k=1)
    idx = np.where(valid)[0]

    mask_dist = np.zeros(len(gaussians), dtype=bool)
    mask_dist[idx] = d < dist

    # ======================
    # BBOX FILTER (IMPORTANT ADDITION)
    # ======================
    inside_bbox = (
        (gaussians[:, 0] >= bbox_min[0]) & (gaussians[:, 0] <= bbox_max[0]) &
        (gaussians[:, 1] >= bbox_min[1]) & (gaussians[:, 1] <= bbox_max[1]) &
        (gaussians[:, 2] >= bbox_min[2]) & (gaussians[:, 2] <= bbox_max[2])
    )

    # ======================
    # FINAL MASK (TOLERANT)
    # ======================
    mask = mask_dist | inside_bbox

    log(f"Kept: {mask.sum()} / {len(mask)}")

    # ======================
    # SAVE
    # ======================
    write_ply(ply, mask, args.out_ply)

    log(f"Saved: {args.out_ply}")


if __name__ == "__main__":
    main()

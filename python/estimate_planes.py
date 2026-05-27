import numpy as np
import struct
import os
import sys
import argparse

# ======================
# LOGGING
# ======================
def log(msg):
    print(f"[INFO] {msg}")

def err(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)

# ======================
# HELP
# ======================
def print_help():
    print("""
COLMAP Scene Inspector

Usage:
  python estimate_planes.py --input /path/to/colmap/sparse/0

Options:
  --input          COLMAP sparse folder (must contain images.bin & points3D.bin)

  --filter         Outlier trimming ratio (default: 0.05)
                   -> 0.0 = no filter
                   -> 0.05 = standard indoor
                   -> 0.1 = strong filtering

  --max-points     Max 3D points loaded (default: 200000)
  --max-cameras    Max cameras loaded (default: 200)

Output:
  NEAR=...
  FAR=...

Example:
  python estimate_planes.py --input ./sparse/0 --filter 0.05
""")
    sys.exit(0)

# ======================
# SAFETY
# ======================
def safe_normalize(q):
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-8:
        return None
    return q / norm

def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    if os.path.getsize(path) < 100:
        raise ValueError(f"File too small / corrupted: {path}")

# ======================
# COLMAP LOADING
# ======================
def read_points3d_bin(path, max_points):
    check_file(path)

    pts = []
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_points):
            f.read(8)
            xyz = struct.unpack("<ddd", f.read(24))
            f.read(3)
            f.read(8)

            track_len = struct.unpack("<Q", f.read(8))[0]
            if track_len > 1_000_000:
                raise ValueError("Corrupted track_len")

            f.read(8 * track_len)

            if np.all(np.isfinite(xyz)):
                pts.append(xyz)

            if len(pts) >= max_points:
                break

    pts = np.array(pts, dtype=np.float64)
    log(f"Loaded points3D: {len(pts)}")
    return pts

def read_images_bin(path, max_cameras):
    check_file(path)

    cams = []

    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        log(f"Images found: {num_images}")

        for _ in range(num_images):
            f.read(4)
            qvec = np.array(struct.unpack("<dddd", f.read(32)))
            tvec = np.array(struct.unpack("<ddd", f.read(24)))
            f.read(4)

            while True:
                if f.read(1) == b"\x00":
                    break

            num_pts2D = struct.unpack("<Q", f.read(8))[0]
            if num_pts2D > 5_000_000:
                raise ValueError("Corrupted num_pts2D")

            f.read(num_pts2D * 24)

            q = safe_normalize(qvec)
            if q is None:
                continue

            qw, qx, qy, qz = q

            R = np.array([
                [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
            ])

            if not np.all(np.isfinite(R)):
                continue

            C = -R.T @ tvec

            if np.all(np.isfinite(C)):
                cams.append(C)

            if len(cams) >= max_cameras:
                break

    cams = np.array(cams, dtype=np.float64)
    log(f"Valid cameras: {len(cams)}")
    return cams

# ======================
# ESTIMATION
# ======================
def estimate_planes(colmap_dir, trim, max_points, max_cameras):

    pts = read_points3d_bin(
        os.path.join(colmap_dir, "points3D.bin"),
        max_points
    )

    cams = read_images_bin(
        os.path.join(colmap_dir, "images.bin"),
        max_cameras
    )

    if len(pts) < 1000 or len(cams) < 2:
        raise RuntimeError("Invalid COLMAP data")

    # ======================
    # FILTER (ROBUST TRIMMING)
    # ======================
    if trim > 0:
        bbox_min = np.percentile(pts, trim * 100, axis=0)
        bbox_max = np.percentile(pts, (1 - trim) * 100, axis=0)
    else:
        bbox_min = np.min(pts, axis=0)
        bbox_max = np.max(pts, axis=0)

    scene_size = np.linalg.norm(bbox_max - bbox_min)

    cam_center = np.mean(cams, axis=0)
    cam_dist = np.linalg.norm(cams - cam_center, axis=1)
    cam_scale = np.median(cam_dist)

    near = cam_scale * 0.3
    far = min(scene_size * 0.6, cam_scale * 2.5)

    near = max(near, 0.05)
    far = max(far, near * 3)

    if not np.isfinite(near) or not np.isfinite(far):
        raise RuntimeError("Invalid result")

    return float(near), float(far)

# ======================
# MAIN
# ======================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--filter", type=float, default=0.05)
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument("--max-cameras", type=int, default=200)

    args, _ = parser.parse_known_args()

    if "--help" in sys.argv:
        print_help()

    if not os.path.isdir(args.input):
        err(f"Invalid input directory: {args.input}")
        sys.exit(1)

    try:
        log("📦 Loading COLMAP scene...")

        near, far = estimate_planes(
            args.input,
            args.filter,
            args.max_points,
            args.max_cameras
        )

        log("📏 Estimated planes:")
        print(f"NEAR={near:.4f}")
        print(f"FAR={far:.4f}")

    except Exception as e:
        err(str(e))
        sys.exit(1)

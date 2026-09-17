#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np


try:
    from scipy.spatial import cKDTree

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(".").resolve()


# =============================================================================
# LOGGING
# =============================================================================

def log(message=""):
    print(message, flush=True)


def warn(message):
    log(f"[WARN] {message}")


def error(message):
    log(f"[ERROR] {message}")


# =============================================================================
# PATHS
# =============================================================================

def dataset_paths(name):
    root = BASE_DIR / name

    return {
        "name": name,
        "root": root,
        "transforms": root / "transforms.json",
        "normalization": root / "transforms_normalization.json",
        "georeferencing": root / "georeferencing.json",
        "sparse_pc": root / "sparse_pc.ply",
        "model_ply": root / "model.ply",
        "colmap_root": root / "colmap",
        "colmap": root / "colmap" / "sparse" / "0",
        "colmap_txt": root / "colmap" / "sparse_txt",
        "colmap_eval_txt": root / "colmap" / "sparse_eval_txt",
        "images": root / "images",
    }


def find_colmap_txt_dir(paths):
    candidates = [
        paths["colmap_txt"],
        paths["colmap_eval_txt"],
        paths["colmap_root"] / "sparse_txt_for_transforms",
        paths["colmap_root"] / "sparse_txt_for_inspection",
    ]

    for candidate in candidates:
        if (
            (candidate / "cameras.txt").exists()
            and (candidate / "images.txt").exists()
            and (candidate / "points3D.txt").exists()
        ):
            return candidate

    return None


# =============================================================================
# FILE DIAGNOSTICS
# =============================================================================

def sha256_file(path, chunk_size=1024 * 1024):
    path = Path(path)

    if not path.exists() or not path.is_file():
        return None

    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def file_info(path, with_hash=False):
    path = Path(path)

    result = {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": None,
        "sha256": None,
    }

    if path.exists() and path.is_file():
        result["size_bytes"] = path.stat().st_size
        if with_hash:
            result["sha256"] = sha256_file(path)

    return result


def print_file_info(label, path, with_hash=False):
    info = file_info(path, with_hash=with_hash)

    if not info["exists"]:
        log(f"{label}: ABSENT -> {path}")
        return info

    log(f"{label}: {path}")
    log(f"  size: {info['size_bytes']} bytes")

    if info["sha256"]:
        log(f"  sha256: {info['sha256']}")

    return info


def print_ply_header(path, max_lines=40):
    path = Path(path)

    if not path.exists():
        warn(f"PLY absent: {path}")
        return

    log(f"\n[PLY HEADER] {path}")

    with open(path, "rb") as f:
        for _ in range(max_lines):
            line = f.readline()

            if not line:
                break

            text = line.decode("ascii", errors="replace").rstrip("\n\r")
            log(f"  {text}")

            if text.strip() == "end_header":
                break


# =============================================================================
# TRANSFORMS.JSON
# =============================================================================

def load_transforms(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"transforms.json absent: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames", [])

    centers = []
    rotations = []
    names = []

    for index, frame in enumerate(frames):
        if "transform_matrix" not in frame:
            warn(f"{path}: frame {index} sans transform_matrix")
            continue

        matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)

        if matrix.shape != (4, 4):
            warn(f"{path}: frame {index}: matrice de forme {matrix.shape}")
            continue

        centers.append(matrix[:3, 3])
        rotations.append(matrix[:3, :3])
        names.append(Path(frame.get("file_path", "")).name)

    if centers:
        centers = np.asarray(centers, dtype=np.float64)
        rotations = np.asarray(rotations, dtype=np.float64)
    else:
        centers = np.zeros((0, 3), dtype=np.float64)
        rotations = np.zeros((0, 3, 3), dtype=np.float64)

    return centers, rotations, names, data

def load_transform_frame_map(transforms_path):
    """
    Charge transforms.json sous forme indexée par nom d'image.

    Retourne :
      metadata
      frames_by_name = {
        image_name: {
          index,
          file_path,
          matrix,
          rotation,
          center,
          colmap_im_id
        }
      }
    """
    transforms_path = Path(transforms_path)

    if not transforms_path.exists():
        raise FileNotFoundError(transforms_path)

    metadata = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = metadata.get("frames", [])

    frames_by_name = {}

    for index, frame in enumerate(frames):
        file_path = frame.get("file_path", "")
        image_name = Path(file_path).name

        if not image_name:
            warn(f"{transforms_path}: frame {index} sans file_path exploitable")
            continue

        if "transform_matrix" not in frame:
            warn(f"{transforms_path}: frame {index} sans transform_matrix")
            continue

        matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)

        if matrix.shape != (4, 4):
            warn(
                f"{transforms_path}: frame {index} matrice invalide: "
                f"{matrix.shape}"
            )
            continue

        if image_name in frames_by_name:
            warn(f"{transforms_path}: image dupliquée dans transforms.json: {image_name}")

        frames_by_name[image_name] = {
            "index": index,
            "file_path": file_path,
            "matrix": matrix,
            "rotation": matrix[:3, :3],
            "center": matrix[:3, 3],
            "colmap_im_id": frame.get("colmap_im_id"),
            "raw": frame,
        }

    return metadata, frames_by_name


def unit_vector(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)

    if norm < 1e-15 or not np.isfinite(norm):
        return np.full_like(vector, np.nan, dtype=np.float64)

    return vector / norm


def angle_between_vectors_deg(a, b):
    a = unit_vector(a)
    b = unit_vector(b)

    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return np.nan

    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def rotation_delta_angle_deg(rotation_a, rotation_b):
    """
    Angle de rotation relatif entre deux matrices R.

    Si R_a et R_b sont très proches, retourne un angle très petit en degrés.
    """
    rotation_a = np.asarray(rotation_a, dtype=np.float64)
    rotation_b = np.asarray(rotation_b, dtype=np.float64)

    if rotation_a.shape != (3, 3) or rotation_b.shape != (3, 3):
        return np.nan

    if not np.isfinite(rotation_a).all() or not np.isfinite(rotation_b).all():
        return np.nan

    delta = rotation_a.T @ rotation_b
    cos_angle = (np.trace(delta) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))

    return float(np.degrees(np.arccos(cos_angle)))


def camera_axes_from_c2w(matrix):
    """
    Extrait les axes caméra depuis une matrice camera-to-world.

    Convention Nerfstudio/OpenGL habituelle :
      - axe X caméra = colonne 0
      - axe Y caméra = colonne 1
      - axe optique / forward = - colonne 2
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    rotation = matrix[:3, :3]

    right = unit_vector(rotation[:, 0])
    up = unit_vector(rotation[:, 1])
    forward = unit_vector(-rotation[:, 2])

    return right, up, forward


def compare_transforms_by_image_name(label_a, path_a, label_b, path_b, top_k=30):
    """
    Compare deux transforms.json image par image.

    C'est plus fort que la comparaison globale des poses :
    - même bbox de caméras ne garantit pas que chaque image a la bonne pose ;
    - même baseline globale ne détecte pas une permutation image/pose ;
    - une petite rotation par image peut tuer les gradients Splatfacto.
    """
    log("\n" + "=" * 90)
    log(f"COMPARAISON FINE transforms.json PAR NOM D'IMAGE: {label_a} VS {label_b}")
    log("=" * 90)

    metadata_a, frames_a = load_transform_frame_map(path_a)
    metadata_b, frames_b = load_transform_frame_map(path_b)

    log("\n--- METADATA ---")

    for key in [
        "w",
        "h",
        "fl_x",
        "fl_y",
        "cx",
        "cy",
        "camera_model",
        "ply_file_path",
        "applied_scale",
    ]:
        value_a = metadata_a.get(key)
        value_b = metadata_b.get(key)
        same = value_a == value_b

        log(
            f"{key}: "
            f"{label_a}={value_a!r} ; "
            f"{label_b}={value_b!r} ; "
            f"same={same}"
        )

    names_a = set(frames_a)
    names_b = set(frames_b)

    common = sorted(names_a & names_b)
    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)

    log("\n--- FRAMES ---")
    log(f"{label_a}: {len(frames_a)} frames")
    log(f"{label_b}: {len(frames_b)} frames")
    log(f"communes: {len(common)}")
    log(f"seulement {label_a}: {len(only_a)}")
    log(f"seulement {label_b}: {len(only_b)}")

    if only_a:
        log(f"premières seulement {label_a}: {only_a[:10]}")

    if only_b:
        log(f"premières seulement {label_b}: {only_b[:10]}")

    center_errors = []
    rotation_angles = []
    right_axis_angles = []
    up_axis_angles = []
    forward_axis_angles = []
    matrix_max_abs_errors = []
    colmap_id_mismatches = []
    order_index_differences = []

    detailed = []

    for name in common:
        frame_a = frames_a[name]
        frame_b = frames_b[name]

        matrix_a = frame_a["matrix"]
        matrix_b = frame_b["matrix"]

        center_error = float(
            np.linalg.norm(frame_a["center"] - frame_b["center"])
        )

        rotation_angle = rotation_delta_angle_deg(
            frame_a["rotation"],
            frame_b["rotation"],
        )

        right_a, up_a, forward_a = camera_axes_from_c2w(matrix_a)
        right_b, up_b, forward_b = camera_axes_from_c2w(matrix_b)

        right_angle = angle_between_vectors_deg(right_a, right_b)
        up_angle = angle_between_vectors_deg(up_a, up_b)
        forward_angle = angle_between_vectors_deg(forward_a, forward_b)

        matrix_max_abs_error = float(np.max(np.abs(matrix_a - matrix_b)))

        id_a = frame_a.get("colmap_im_id")
        id_b = frame_b.get("colmap_im_id")

        if id_a != id_b:
            colmap_id_mismatches.append((name, id_a, id_b))

        order_index_differences.append(
            int(frame_a["index"]) - int(frame_b["index"])
        )

        center_errors.append(center_error)
        rotation_angles.append(rotation_angle)
        right_axis_angles.append(right_angle)
        up_axis_angles.append(up_angle)
        forward_axis_angles.append(forward_angle)
        matrix_max_abs_errors.append(matrix_max_abs_error)

        detailed.append(
            {
                "name": name,
                "center_error": center_error,
                "rotation_angle_deg": rotation_angle,
                "right_axis_angle_deg": right_angle,
                "up_axis_angle_deg": up_angle,
                "forward_axis_angle_deg": forward_angle,
                "matrix_max_abs_error": matrix_max_abs_error,
                "index_a": frame_a["index"],
                "index_b": frame_b["index"],
                "colmap_im_id_a": id_a,
                "colmap_im_id_b": id_b,
            }
        )

    log("\n--- ERREURS PAR NOM D'IMAGE ---")
    log(f"center error: {stats_1d(center_errors)}")
    log(f"rotation angle deg: {stats_1d(rotation_angles)}")
    log(f"right axis angle deg: {stats_1d(right_axis_angles)}")
    log(f"up axis angle deg: {stats_1d(up_axis_angles)}")
    log(f"forward/view axis angle deg: {stats_1d(forward_axis_angles)}")
    log(f"matrix max abs error: {stats_1d(matrix_max_abs_errors)}")
    log(f"index difference stats: {stats_1d(order_index_differences)}")
    log(f"colmap_im_id mismatches: {len(colmap_id_mismatches)}")

    if colmap_id_mismatches:
        log("premiers colmap_im_id mismatches:")
        for item in colmap_id_mismatches[:top_k]:
            name, id_a, id_b = item
            log(f"  {name}: {label_a}={id_a}, {label_b}={id_b}")

    log(f"\n--- TOP {top_k} ROTATION DIFF ---")

    detailed_by_rotation = sorted(
        detailed,
        key=lambda item: (
            -1.0
            if not np.isfinite(item["rotation_angle_deg"])
            else item["rotation_angle_deg"]
        ),
        reverse=True,
    )

    for item in detailed_by_rotation[:top_k]:
        log(
            f"{item['name']}: "
            f"rot={item['rotation_angle_deg']:.9f} deg, "
            f"forward={item['forward_axis_angle_deg']:.9f} deg, "
            f"up={item['up_axis_angle_deg']:.9f} deg, "
            f"center={item['center_error']:.9f} m, "
            f"idx {label_a}/{label_b}={item['index_a']}/{item['index_b']}, "
            f"colmap_id {label_a}/{label_b}="
            f"{item['colmap_im_id_a']}/{item['colmap_im_id_b']}"
        )

    log(f"\n--- TOP {top_k} CENTER DIFF ---")

    detailed_by_center = sorted(
        detailed,
        key=lambda item: item["center_error"],
        reverse=True,
    )

    for item in detailed_by_center[:top_k]:
        log(
            f"{item['name']}: "
            f"center={item['center_error']:.9f} m, "
            f"rot={item['rotation_angle_deg']:.9f} deg, "
            f"forward={item['forward_axis_angle_deg']:.9f} deg"
        )

    return {
        "common": common,
        "only_a": only_a,
        "only_b": only_b,
        "center_error": stats_1d(center_errors),
        "rotation_angle_deg": stats_1d(rotation_angles),
        "right_axis_angle_deg": stats_1d(right_axis_angles),
        "up_axis_angle_deg": stats_1d(up_axis_angles),
        "forward_axis_angle_deg": stats_1d(forward_axis_angles),
        "matrix_max_abs_error": stats_1d(matrix_max_abs_errors),
        "colmap_id_mismatches": colmap_id_mismatches,
        "details": detailed,
    }


def detect_pose_permutation_by_nearest_center(label_a, path_a, label_b, path_b):
    """
    Détecte si les mêmes poses existent mais sont associées aux mauvais noms d'images.

    Pour chaque image de B, on cherche la pose de A dont le centre caméra
    est le plus proche. Si le nom ne correspond pas, il y a probablement
    une permutation image -> pose.
    """
    log("\n" + "=" * 90)
    log(f"DETECTION PERMUTATION IMAGE -> POSE: {label_a} VS {label_b}")
    log("=" * 90)

    _, frames_a = load_transform_frame_map(path_a)
    _, frames_b = load_transform_frame_map(path_b)

    if not frames_a or not frames_b:
        warn("Impossible: frames vides")
        return {}

    names_a = sorted(frames_a)
    centers_a = np.asarray(
        [frames_a[name]["center"] for name in names_a],
        dtype=np.float64,
    )

    permutation_suspects = []
    nearest_distances = []

    for name_b, frame_b in sorted(frames_b.items()):
        center_b = frame_b["center"]

        distances = np.linalg.norm(centers_a - center_b, axis=1)
        nearest_index = int(np.argmin(distances))
        nearest_name_a = names_a[nearest_index]
        nearest_distance = float(distances[nearest_index])

        nearest_distances.append(nearest_distance)

        if nearest_name_a != name_b:
            permutation_suspects.append(
                {
                    "image_b": name_b,
                    "nearest_image_a": nearest_name_a,
                    "distance": nearest_distance,
                    "index_b": frame_b["index"],
                    "index_a": frames_a[nearest_name_a]["index"],
                }
            )

    log(f"nearest center distance stats: {stats_1d(nearest_distances)}")
    log(f"suspicions permutation: {len(permutation_suspects)} / {len(frames_b)}")

    for item in permutation_suspects[:30]:
        log(
            f"{label_b}:{item['image_b']} "
            f"semble correspondre à {label_a}:{item['nearest_image_a']} "
            f"distance={item['distance']:.9e} "
            f"indices {label_b}/{label_a}={item['index_b']}/{item['index_a']}"
        )

    return {
        "nearest_center_distance": stats_1d(nearest_distances),
        "permutation_suspects": permutation_suspects,
    }
    
def project_points_with_transform_conventions(points, frame, metadata):
    """
    Projette les points 3D avec deux conventions possibles.

    Convention A: OpenGL / Nerfstudio classique
      caméra regarde vers -Z
      depth = -z_cam
      u = fx * x/depth + cx
      v = fy * (-y)/depth + cy

    Convention B: OpenCV/COLMAP classique
      caméra regarde vers +Z
      depth = z_cam
      u = fx * x/depth + cx
      v = fy * y/depth + cy

    On retourne les deux pour diagnostic relatif.
    """
    matrix_c2w = np.asarray(frame["matrix"], dtype=np.float64)
    world_to_camera = np.linalg.inv(matrix_c2w)

    points = np.asarray(points, dtype=np.float64)

    points_h = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)],
        axis=1,
    )

    camera_points = (world_to_camera @ points_h.T).T[:, :3]

    fx = float(metadata["fl_x"])
    fy = float(metadata["fl_y"])
    cx = float(metadata["cx"])
    cy = float(metadata["cy"])
    width = int(metadata["w"])
    height = int(metadata["h"])

    x = camera_points[:, 0]
    y = camera_points[:, 1]
    z = camera_points[:, 2]

    result = {}

    # Convention OpenGL / Nerfstudio
    depth_gl = -z
    valid_gl = np.isfinite(camera_points).all(axis=1) & (depth_gl > 1e-9)

    u_gl = np.full((len(points),), np.nan, dtype=np.float64)
    v_gl = np.full((len(points),), np.nan, dtype=np.float64)

    u_gl[valid_gl] = fx * (x[valid_gl] / depth_gl[valid_gl]) + cx
    v_gl[valid_gl] = fy * (-y[valid_gl] / depth_gl[valid_gl]) + cy

    inside_gl = (
        valid_gl
        & (u_gl >= 0.0)
        & (u_gl < width)
        & (v_gl >= 0.0)
        & (v_gl < height)
    )

    result["opengl_minus_z"] = {
        "valid": valid_gl,
        "inside": inside_gl,
        "depth": depth_gl,
        "u": u_gl,
        "v": v_gl,
    }

    # Convention OpenCV / COLMAP
    depth_cv = z
    valid_cv = np.isfinite(camera_points).all(axis=1) & (depth_cv > 1e-9)

    u_cv = np.full((len(points),), np.nan, dtype=np.float64)
    v_cv = np.full((len(points),), np.nan, dtype=np.float64)

    u_cv[valid_cv] = fx * (x[valid_cv] / depth_cv[valid_cv]) + cx
    v_cv[valid_cv] = fy * (y[valid_cv] / depth_cv[valid_cv]) + cy

    inside_cv = (
        valid_cv
        & (u_cv >= 0.0)
        & (u_cv < width)
        & (v_cv >= 0.0)
        & (v_cv < height)
    )

    result["opencv_plus_z"] = {
        "valid": valid_cv,
        "inside": inside_cv,
        "depth": depth_cv,
        "u": u_cv,
        "v": v_cv,
    }

    return result


def sparse_pc_visibility_from_transforms(
    dataset_label,
    transforms_path,
    sparse_pc_path,
    max_points=200000,
):
    """
    Mesure combien de points du sparse_pc.ply sont visibles dans les caméras
    définies par transforms.json.

    C'est un test critique pour ton problème :
    si drone8c voit beaucoup de points dans les images mais drone15d presque aucun,
    alors le problème est dans transforms.json, même si le PLY est identique.
    """
    log("\n" + "=" * 90)
    log(f"VISIBILITE sparse_pc.ply VIA transforms.json: {dataset_label}")
    log("=" * 90)

    metadata, frames_by_name = load_transform_frame_map(transforms_path)

    points, ply_metadata = load_ply_points(sparse_pc_path)

    if len(points) == 0:
        warn(f"{dataset_label}: aucun point dans sparse_pc.ply")
        return {}

    points = points[finite_rows(points)]

    original_count = len(points)

    if max_points is not None and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points = points[indices]

    log(f"points PLY total: {original_count}")
    log(f"points testés   : {len(points)}")
    log(f"frames          : {len(frames_by_name)}")

    conventions = ["opengl_minus_z", "opencv_plus_z"]

    global_stats = {
        convention: {
            "visible_counts_per_point": np.zeros((len(points),), dtype=np.int32),
            "inside_counts_per_image": [],
            "valid_counts_per_image": [],
            "depths_inside": [],
        }
        for convention in conventions
    }

    per_image_rows = []

    for image_name, frame in sorted(frames_by_name.items()):
        projections = project_points_with_transform_conventions(
            points,
            frame,
            metadata,
        )

        row = {
            "image_name": image_name,
        }

        for convention in conventions:
            valid = projections[convention]["valid"]
            inside = projections[convention]["inside"]
            depth = projections[convention]["depth"]

            valid_count = int(np.count_nonzero(valid))
            inside_count = int(np.count_nonzero(inside))

            global_stats[convention]["valid_counts_per_image"].append(valid_count)
            global_stats[convention]["inside_counts_per_image"].append(inside_count)
            global_stats[convention]["visible_counts_per_point"][inside] += 1

            if inside_count:
                global_stats[convention]["depths_inside"].extend(
                    depth[inside].tolist()
                )

            row[f"{convention}_valid"] = valid_count
            row[f"{convention}_inside"] = inside_count

        per_image_rows.append(row)

    result = {}

    for convention in conventions:
        visible_counts = global_stats[convention]["visible_counts_per_point"]
        inside_counts_per_image = np.asarray(
            global_stats[convention]["inside_counts_per_image"],
            dtype=np.float64,
        )
        valid_counts_per_image = np.asarray(
            global_stats[convention]["valid_counts_per_image"],
            dtype=np.float64,
        )
        depths_inside = np.asarray(
            global_stats[convention]["depths_inside"],
            dtype=np.float64,
        )

        points_seen_at_least_1 = int(np.count_nonzero(visible_counts >= 1))
        points_seen_at_least_2 = int(np.count_nonzero(visible_counts >= 2))
        points_seen_at_least_5 = int(np.count_nonzero(visible_counts >= 5))

        log(f"\n--- Convention {convention} ---")
        log(f"valid points per image: {stats_1d(valid_counts_per_image)}")
        log(f"inside points per image: {stats_1d(inside_counts_per_image)}")
        log(f"point visibility count stats: {stats_1d(visible_counts)}")
        log(
            f"points vus >=1 image: "
            f"{points_seen_at_least_1}/{len(points)} "
            f"({100.0 * points_seen_at_least_1 / max(1, len(points)):.3f}%)"
        )
        log(
            f"points vus >=2 images: "
            f"{points_seen_at_least_2}/{len(points)} "
            f"({100.0 * points_seen_at_least_2 / max(1, len(points)):.3f}%)"
        )
        log(
            f"points vus >=5 images: "
            f"{points_seen_at_least_5}/{len(points)} "
            f"({100.0 * points_seen_at_least_5 / max(1, len(points)):.3f}%)"
        )
        log(f"depths for inside projections: {stats_1d(depths_inside)}")

        worst_images = sorted(
            per_image_rows,
            key=lambda row: row[f"{convention}_inside"],
        )[:10]

        log("10 images avec le moins de points projetés dedans:")
        for row in worst_images:
            log(
                f"  {row['image_name']}: "
                f"inside={row[f'{convention}_inside']} "
                f"valid={row[f'{convention}_valid']}"
            )

        result[convention] = {
            "points_tested": int(len(points)),
            "valid_points_per_image": stats_1d(valid_counts_per_image),
            "inside_points_per_image": stats_1d(inside_counts_per_image),
            "point_visibility_count": stats_1d(visible_counts),
            "points_seen_at_least_1": points_seen_at_least_1,
            "points_seen_at_least_2": points_seen_at_least_2,
            "points_seen_at_least_5": points_seen_at_least_5,
            "depths_inside": stats_1d(depths_inside),
        }

    return result


def compare_sparse_pc_visibility(
    label_a,
    visibility_a,
    label_b,
    visibility_b,
):
    log("\n" + "=" * 90)
    log(f"COMPARAISON VISIBILITE sparse_pc.ply: {label_a} VS {label_b}")
    log("=" * 90)

    for convention in ["opengl_minus_z", "opencv_plus_z"]:
        a = visibility_a.get(convention, {})
        b = visibility_b.get(convention, {})

        log(f"\n--- Convention {convention} ---")

        for key in [
            "points_tested",
            "points_seen_at_least_1",
            "points_seen_at_least_2",
            "points_seen_at_least_5",
            "inside_points_per_image",
            "valid_points_per_image",
            "point_visibility_count",
            "depths_inside",
        ]:
            log(
                f"{key}: "
                f"{label_a}={a.get(key)} ; "
                f"{label_b}={b.get(key)}"
            )


# =============================================================================
# GEOMETRY
# =============================================================================

def quaternion_to_rotmat(q):
    q = np.asarray(q, dtype=np.float64)

    if q.shape != (4,):
        raise ValueError(f"Quaternion invalide: {q.shape}")

    norm = np.linalg.norm(q)

    if norm < 1e-15:
        raise ValueError("Quaternion nul")

    qw, qx, qy, qz = q / norm

    return np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qw * qz),
                2 * (qx * qz + qw * qy),
            ],
            [
                2 * (qx * qy + qw * qz),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qw * qx),
            ],
            [
                2 * (qx * qz - qw * qy),
                2 * (qy * qz + qw * qx),
                1 - 2 * (qx * qx + qz * qz),
            ],
        ],
        dtype=np.float64,
    )


def translation_to_center(rotation_world_to_camera, translation_world_to_camera):
    return -rotation_world_to_camera.T @ translation_world_to_camera


def finite_rows(array):
    array = np.asarray(array)

    if array.ndim == 1:
        return np.isfinite(array)

    return np.isfinite(array).all(axis=1)


def stats_1d(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    return {
        "n": int(len(values)),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def bbox_stats(points):
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points attendus sous forme (N,3), reçu {points.shape}")

    if len(points) == 0:
        return {
            "n": 0,
            "n_finite": 0,
            "n_invalid": 0,
            "min": None,
            "max": None,
            "mean": None,
            "diag": None,
        }

    finite_mask = finite_rows(points)
    valid = points[finite_mask]

    if len(valid) == 0:
        return {
            "n": int(len(points)),
            "n_finite": 0,
            "n_invalid": int(len(points)),
            "min": None,
            "max": None,
            "mean": None,
            "diag": None,
        }

    minimum = valid.min(axis=0)
    maximum = valid.max(axis=0)

    return {
        "n": int(len(points)),
        "n_finite": int(len(valid)),
        "n_invalid": int(len(points) - len(valid)),
        "min": minimum,
        "max": maximum,
        "mean": valid.mean(axis=0),
        "diag": float(np.linalg.norm(maximum - minimum)),
    }


def print_bbox_stats(prefix, points):
    stats = bbox_stats(points)

    log(f"{prefix}: n={stats['n']} finite={stats['n_finite']} invalid={stats['n_invalid']}")

    if stats["n_finite"] == 0:
        log("  bbox: aucun point fini")
        return stats

    log(f"  bbox min: {stats['min']}")
    log(f"  bbox max: {stats['max']}")
    log(f"  bbox diag: {stats['diag']:.9f}")
    log(f"  mean: {stats['mean']}")

    return stats


# =============================================================================
# POSE STATISTICS
# =============================================================================

def pairwise_baselines(centers):
    centers = np.asarray(centers, dtype=np.float64)

    if len(centers) < 2:
        return np.zeros((0,), dtype=np.float64)

    distances = []

    for i in range(len(centers)):
        delta = centers[i + 1 :] - centers[i]
        distances.extend(np.linalg.norm(delta, axis=1))

    return np.asarray(distances, dtype=np.float64)


def consecutive_baselines(centers):
    centers = np.asarray(centers, dtype=np.float64)

    if len(centers) < 2:
        return np.zeros((0,), dtype=np.float64)

    return np.linalg.norm(centers[1:] - centers[:-1], axis=1)


def viewing_dirs_from_c2w(rotations):
    rotations = np.asarray(rotations, dtype=np.float64)

    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        return np.zeros((0, 3), dtype=np.float64)

    # Convention Nerfstudio/OpenGL utilisée par le script initial:
    # direction de vue = - troisième colonne de R_c2w.
    return -rotations[:, :, 2]


def angular_spread(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)

    if len(vectors) == 0:
        return np.zeros((0,), dtype=np.float64)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-15)

    mean_vector = normalized.mean(axis=0)
    mean_norm = np.linalg.norm(mean_vector)

    if mean_norm < 1e-15:
        return np.full((len(vectors),), np.nan)

    mean_vector /= mean_norm

    return np.degrees(
        np.arccos(
            np.clip(normalized @ mean_vector, -1.0, 1.0)
        )
    )


def orthonormality_stats(rotations):
    rotations = np.asarray(rotations, dtype=np.float64)

    if len(rotations) == 0:
        return None

    errors = []
    determinants = []

    for rotation in rotations:
        errors.append(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
        determinants.append(np.linalg.det(rotation))

    return {
        "orth_mean": float(np.mean(errors)),
        "orth_p95": float(np.percentile(errors, 95)),
        "det_mean": float(np.mean(determinants)),
        "det_min": float(np.min(determinants)),
        "det_max": float(np.max(determinants)),
    }


def summarize_pose_set(label, centers, rotations, metadata):
    log(f"\n=== {label}: POSES ===")
    log(f"frames: {len(centers)}")

    if len(centers) == 0:
        warn("Aucune pose")
        return {}

    finite_center_mask = finite_rows(centers)
    finite_rotation_mask = np.isfinite(rotations).all(axis=(1, 2))

    log(f"finite centers: {np.count_nonzero(finite_center_mask)}/{len(centers)}")
    log(f"finite rotations: {np.count_nonzero(finite_rotation_mask)}/{len(rotations)}")

    center_stats = bbox_stats(centers)
    print_bbox_stats("camera centers", centers)

    pairwise = pairwise_baselines(centers)
    consecutive = consecutive_baselines(centers)

    if len(pairwise):
        log(
            "pairwise baseline min/mean/p50/p95/max: "
            f"{np.min(pairwise):.9f} / "
            f"{np.mean(pairwise):.9f} / "
            f"{np.percentile(pairwise, 50):.9f} / "
            f"{np.percentile(pairwise, 95):.9f} / "
            f"{np.max(pairwise):.9f}"
        )

    if len(consecutive):
        log(
            "consecutive baseline min/mean/p50/p95/max: "
            f"{np.min(consecutive):.9f} / "
            f"{np.mean(consecutive):.9f} / "
            f"{np.percentile(consecutive, 50):.9f} / "
            f"{np.percentile(consecutive, 95):.9f} / "
            f"{np.max(consecutive):.9f}"
        )

    directions = viewing_dirs_from_c2w(rotations)
    angles = angular_spread(directions)
    angles = angles[np.isfinite(angles)]

    if len(angles):
        log(
            "view angle spread to mean min/mean/p50/p95/max deg: "
            f"{np.min(angles):.6f} / "
            f"{np.mean(angles):.6f} / "
            f"{np.percentile(angles, 50):.6f} / "
            f"{np.percentile(angles, 95):.6f} / "
            f"{np.max(angles):.6f}"
        )

    rotation_stats = orthonormality_stats(rotations)

    if rotation_stats:
        log(
            "rotation orthonormality mean/p95: "
            f"{rotation_stats['orth_mean']:.3e} / "
            f"{rotation_stats['orth_p95']:.3e}"
        )
        log(
            "rotation det min/mean/max: "
            f"{rotation_stats['det_min']:.9f} / "
            f"{rotation_stats['det_mean']:.9f} / "
            f"{rotation_stats['det_max']:.9f}"
        )

    log(f"camera_model: {metadata.get('camera_model')}")
    log(f"w,h: {metadata.get('w')}x{metadata.get('h')}")
    log(f"fl_x/fl_y: {metadata.get('fl_x')} / {metadata.get('fl_y')}")
    log(f"cx/cy: {metadata.get('cx')} / {metadata.get('cy')}")
    log(f"ply_file_path: {metadata.get('ply_file_path')}")

    return {
        "center_bbox": center_stats,
        "pairwise": stats_1d(pairwise),
        "consecutive": stats_1d(consecutive),
        "view_angles": stats_1d(angles),
        "rotation": rotation_stats,
    }


# =============================================================================
# COLMAP IMAGES.TXT
# =============================================================================

def load_colmap_images_txt(images_txt):
    """
    Charge images.txt COLMAP.

    Retourne :
      {
        image_name: {
          image_id,
          camera_id,
          q,
          t,
          R,
          C,
          points2d: [(x, y, point3d_id), ...]
        }
      }

    Attention : la deuxième ligne d'une image contient les points 2D.
    """

    images_txt = Path(images_txt)

    if not images_txt.exists():
        raise FileNotFoundError(images_txt)

    lines = images_txt.read_text(encoding="utf-8").splitlines()
    result = {}

    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line or line.startswith("#"):
            i += 1
            continue

        tokens = line.split()

        if len(tokens) < 10:
            i += 1
            continue

        image_id = int(tokens[0])

        q = np.array(
            [
                float(tokens[1]),
                float(tokens[2]),
                float(tokens[3]),
                float(tokens[4]),
            ],
            dtype=np.float64,
        )

        t = np.array(
            [
                float(tokens[5]),
                float(tokens[6]),
                float(tokens[7]),
            ],
            dtype=np.float64,
        )

        camera_id = int(tokens[8])
        image_name = " ".join(tokens[9:])

        rotation = quaternion_to_rotmat(q)
        center = translation_to_center(rotation, t)

        points2d = []

        if i + 1 < len(lines):
            points_line = lines[i + 1].strip()

            if points_line and not points_line.startswith("#"):
                point_tokens = points_line.split()

                for j in range(0, len(point_tokens) - 2, 3):
                    x = float(point_tokens[j])
                    y = float(point_tokens[j + 1])
                    point3d_id = int(point_tokens[j + 2])

                    points2d.append((x, y, point3d_id))

                i += 2
            else:
                i += 1
        else:
            i += 1

        result[Path(image_name).name] = {
            "image_id": image_id,
            "camera_id": camera_id,
            "q": q,
            "t": t,
            "R": rotation,
            "C": center,
            "name": Path(image_name).name,
            "points2d": points2d,
        }

    return result


# =============================================================================
# COLMAP CAMERAS.TXT
# =============================================================================

def load_colmap_cameras_txt(cameras_txt):
    cameras_txt = Path(cameras_txt)

    if not cameras_txt.exists():
        raise FileNotFoundError(cameras_txt)

    result = {}

    for line in cameras_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        tokens = line.split()

        if len(tokens) < 5:
            continue

        camera_id = int(tokens[0])
        model = tokens[1]
        width = int(tokens[2])
        height = int(tokens[3])
        params = np.asarray(list(map(float, tokens[4:])), dtype=np.float64)

        result[camera_id] = {
            "camera_id": camera_id,
            "model": model,
            "w": width,
            "h": height,
            "params": params,
        }

    return result


def camera_intrinsics(camera):
    model = camera["model"]
    params = camera["params"]

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        return {
            "fx": f,
            "fy": f,
            "cx": cx,
            "cy": cy,
            "distortion": params[3:],
        }

    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        return {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "distortion": params[4:],
        }

    if model in ("SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = params[:3]
        return {
            "fx": f,
            "fy": f,
            "cx": cx,
            "cy": cy,
            "distortion": params[3:],
        }

    if model in ("OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = params[:4]
        return {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "distortion": params[4:],
        }

    if model == "FOV":
        f, cx, cy, omega = params[:4]
        return {
            "fx": f,
            "fy": f,
            "cx": cx,
            "cy": cy,
            "distortion": np.asarray([omega]),
        }

    return {
        "fx": np.nan,
        "fy": np.nan,
        "cx": np.nan,
        "cy": np.nan,
        "distortion": params,
    }


def summarize_cameras(label, cameras):
    log(f"\n=== {label}: CAMERAS ===")

    if not cameras:
        warn("Aucune caméra COLMAP")
        return {}

    for camera_id, camera in cameras.items():
        intrinsics = camera_intrinsics(camera)

        log(
            f"camera_id={camera_id} "
            f"model={camera['model']} "
            f"size={camera['w']}x{camera['h']} "
            f"params={camera['params']}"
        )

        log(
            "  fx/fy/cx/cy="
            f"{intrinsics['fx']:.9f}/"
            f"{intrinsics['fy']:.9f}/"
            f"{intrinsics['cx']:.9f}/"
            f"{intrinsics['cy']:.9f}"
        )

        if len(intrinsics["distortion"]):
            log(f"  distortion={intrinsics['distortion']}")

    return cameras


# =============================================================================
# COLMAP POINTS3D.TXT
# =============================================================================

def load_colmap_points3d_txt(points_txt):
    """
    Retourne :

      xyz       (N,3)
      rgb       (N,3)
      errors    (N,)
      track_len (N,)
      point_ids (N,)
      tracks    list[list[(image_id, point2d_idx)]]
    """

    points_txt = Path(points_txt)

    if not points_txt.exists():
        raise FileNotFoundError(points_txt)

    xyz = []
    rgb = []
    errors = []
    track_lengths = []
    point_ids = []
    tracks = []

    for line in points_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        tokens = line.split()

        if len(tokens) < 8:
            continue

        point_id = int(tokens[0])

        point_xyz = [
            float(tokens[1]),
            float(tokens[2]),
            float(tokens[3]),
        ]

        point_rgb = [
            float(tokens[4]),
            float(tokens[5]),
            float(tokens[6]),
        ]

        error = float(tokens[7])

        track_tokens = tokens[8:]
        point_track = []

        for i in range(0, len(track_tokens) - 1, 2):
            image_id = int(track_tokens[i])
            point2d_idx = int(track_tokens[i + 1])
            point_track.append((image_id, point2d_idx))

        point_ids.append(point_id)
        xyz.append(point_xyz)
        rgb.append(point_rgb)
        errors.append(error)
        track_lengths.append(len(point_track))
        tracks.append(point_track)

    if not xyz:
        return {
            "xyz": np.zeros((0, 3), dtype=np.float64),
            "rgb": np.zeros((0, 3), dtype=np.float64),
            "errors": np.zeros((0,), dtype=np.float64),
            "track_lengths": np.zeros((0,), dtype=np.int64),
            "point_ids": np.zeros((0,), dtype=np.int64),
            "tracks": [],
        }

    return {
        "xyz": np.asarray(xyz, dtype=np.float64),
        "rgb": np.asarray(rgb, dtype=np.float64),
        "errors": np.asarray(errors, dtype=np.float64),
        "track_lengths": np.asarray(track_lengths, dtype=np.int64),
        "point_ids": np.asarray(point_ids, dtype=np.int64),
        "tracks": tracks,
    }


def summarize_points3d(label, points_data):
    xyz = points_data["xyz"]
    errors = points_data["errors"]
    track_lengths = points_data["track_lengths"]

    log(f"\n=== {label}: POINTS3D ===")

    print_bbox_stats("points", xyz)

    if len(errors):
        log(f"reprojection error stats: {stats_1d(errors)}")

    if len(track_lengths):
        log(f"track length stats: {stats_1d(track_lengths)}")

        for threshold in [1, 2, 3, 4, 5, 10]:
            count = np.count_nonzero(track_lengths >= threshold)
            log(
                f"tracks >= {threshold}: "
                f"{count}/{len(track_lengths)} "
                f"({100.0 * count / len(track_lengths):.3f}%)"
            )

    invalid_xyz = ~finite_rows(xyz)

    if np.any(invalid_xyz):
        indices = np.flatnonzero(invalid_xyz)
        warn(
            f"{label}: {len(indices)} points avec NaN/Inf; "
            f"premiers indices={indices[:20].tolist()}"
        )

    return {
        "bbox": bbox_stats(xyz),
        "reprojection_error": stats_1d(errors),
        "track_length": stats_1d(track_lengths),
    }


# =============================================================================
# TRACKS / VISIBILITY
# =============================================================================

def summarize_image_tracks(label, images):
    log(f"\n=== {label}: IMAGE TRACKS ===")

    if not images:
        warn("Aucune image")
        return {}

    observations = []
    observed_3d = []
    per_image = []

    for image_name, image in sorted(images.items()):
        n_obs = len(image["points2d"])
        n_3d = sum(point3d_id >= 0 for _, _, point3d_id in image["points2d"])

        observations.append(n_obs)
        observed_3d.append(n_3d)

        per_image.append(
            {
                "name": image_name,
                "n_observations": n_obs,
                "n_observed_3d": n_3d,
            }
        )

        log(
            f"{image_name}: "
            f"observations={n_obs}, "
            f"with_3d_id={n_3d}"
        )

    observations = np.asarray(observations, dtype=np.float64)
    observed_3d = np.asarray(observed_3d, dtype=np.float64)

    log(f"observations stats: {stats_1d(observations)}")
    log(f"observed 3D stats: {stats_1d(observed_3d)}")

    return {
        "observations": stats_1d(observations),
        "observed_3d": stats_1d(observed_3d),
        "per_image": per_image,
    }


def build_image_id_map(images):
    return {
        data["image_id"]: data
        for data in images.values()
    }


def track_consistency(points_data, images):
    """
    Vérifie la cohérence entre les tracks de points3D.txt
    et les points 2D des images.txt.
    """

    image_by_id = build_image_id_map(images)

    total_tracks = 0
    missing_images = 0
    bad_point2d_indices = 0

    for track in points_data["tracks"]:
        for image_id, point2d_idx in track:
            total_tracks += 1

            if image_id not in image_by_id:
                missing_images += 1
                continue

            points2d = image_by_id[image_id]["points2d"]

            if point2d_idx < 0 or point2d_idx >= len(points2d):
                bad_point2d_indices += 1

    return {
        "total_tracks": total_tracks,
        "missing_images": missing_images,
        "bad_point2d_indices": bad_point2d_indices,
    }


# =============================================================================
# PROJECTION / PROFONDEUR / REPROJECTION
# =============================================================================

def project_point(point_world, image_data, camera):
    rotation = image_data["R"]
    translation = image_data["t"]

    point_camera = rotation @ point_world + translation

    x, y, z = point_camera

    if not np.isfinite(point_camera).all():
        return point_camera, np.array([np.nan, np.nan]), False

    if z <= 1e-12:
        return point_camera, np.array([np.nan, np.nan]), False

    intrinsics = camera_intrinsics(camera)

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    if not np.isfinite([fx, fy, cx, cy]).all():
        return point_camera, np.array([np.nan, np.nan]), False

    xn = x / z
    yn = y / z

    model = camera["model"]
    params = camera["params"]

    xd = xn
    yd = yn

    if model in ("SIMPLE_RADIAL", "RADIAL"):
        k1 = params[3] if len(params) > 3 else 0.0
        r2 = xn * xn + yn * yn
        radial = 1.0 + k1 * r2

        if model == "RADIAL":
            k2 = params[4] if len(params) > 4 else 0.0
            k3 = params[5] if len(params) > 5 else 0.0
            radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2

        xd = xn * radial
        yd = yn * radial

    elif model in ("OPENCV", "FULL_OPENCV"):
        k1 = params[4] if len(params) > 4 else 0.0
        k2 = params[5] if len(params) > 5 else 0.0
        p1 = params[6] if len(params) > 6 else 0.0
        p2 = params[7] if len(params) > 7 else 0.0

        r2 = xn * xn + yn * yn
        radial = 1.0 + k1 * r2 + k2 * r2 * r2

        xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn

    pixel = np.array(
        [
            fx * xd + cx,
            fy * yd + cy,
        ],
        dtype=np.float64,
    )

    return point_camera, pixel, True


def compute_reprojection_statistics(points_data, images, cameras):
    """
    Reprojette les points3D sur les observations de images.txt.

    Important:
    - Pour FULL_OPENCV, seuls les quatre premiers paramètres de distortion
      sont appliqués ici.
    - Les résultats servent au diagnostic relatif entre datasets.
    """

    image_by_id = build_image_id_map(images)

    errors = []
    depths = []
    behind = 0
    missing_camera = 0
    invalid_observations = 0
    tested_observations = 0
    projected_inside = 0
    projected_outside = 0

    xyz = points_data["xyz"]

    for point_index, track in enumerate(points_data["tracks"]):
        if point_index >= len(xyz):
            break

        point = xyz[point_index]

        if not np.isfinite(point).all():
            invalid_observations += len(track)
            continue

        for image_id, point2d_idx in track:
            if image_id not in image_by_id:
                missing_camera += 1
                continue

            image = image_by_id[image_id]
            camera_id = image["camera_id"]

            if camera_id not in cameras:
                missing_camera += 1
                continue

            if point2d_idx < 0 or point2d_idx >= len(image["points2d"]):
                invalid_observations += 1
                continue

            observed_xy = np.asarray(
                image["points2d"][point2d_idx][:2],
                dtype=np.float64,
            )

            camera_point, projected_xy, valid = project_point(
                point,
                image,
                cameras[camera_id],
            )

            if not valid:
                if np.isfinite(camera_point).all() and camera_point[2] <= 0:
                    behind += 1
                continue

            tested_observations += 1
            depths.append(camera_point[2])

            reproj_error = np.linalg.norm(projected_xy - observed_xy)

            if np.isfinite(reproj_error):
                errors.append(reproj_error)

            width = cameras[camera_id]["w"]
            height = cameras[camera_id]["h"]

            if 0 <= projected_xy[0] < width and 0 <= projected_xy[1] < height:
                projected_inside += 1
            else:
                projected_outside += 1

    result = {
        "tested_observations": tested_observations,
        "missing_camera": missing_camera,
        "invalid_observations": invalid_observations,
        "behind_camera": behind,
        "projected_inside": projected_inside,
        "projected_outside": projected_outside,
        "reprojection_error_px": stats_1d(errors),
        "depth": stats_1d(depths),
    }

    return result


def print_reprojection_statistics(label, statistics):
    log(f"\n=== {label}: PROJECTION / REPROJECTION ===")
    log(f"tested observations: {statistics['tested_observations']}")
    log(f"missing camera: {statistics['missing_camera']}")
    log(f"invalid observations: {statistics['invalid_observations']}")
    log(f"behind camera: {statistics['behind_camera']}")
    log(f"projected inside image: {statistics['projected_inside']}")
    log(f"projected outside image: {statistics['projected_outside']}")
    log(f"depth stats: {statistics['depth']}")
    log(f"reprojection error px: {statistics['reprojection_error_px']}")


# =============================================================================
# POINT CLOUD DENSITY / NEAREST NEIGHBOURS
# =============================================================================

def nearest_neighbor_statistics(points, label):
    points = np.asarray(points, dtype=np.float64)
    finite_mask = finite_rows(points)
    points = points[finite_mask]

    log(f"\n=== {label}: NEAREST NEIGHBOUR DENSITY ===")

    if len(points) < 2:
        warn("Pas assez de points pour calculer les voisins")
        return {}

    if not HAVE_SCIPY:
        warn("scipy absent: impossible de calculer les voisins rapidement")
        warn("Installation: python -m pip install scipy")
        return {}

    tree = cKDTree(points)
    distances, indices = tree.query(points, k=2)

    nearest = distances[:, 1]

    duplicate_count = int(np.count_nonzero(nearest <= 1e-12))

    result = {
        "nearest_neighbor": stats_1d(nearest),
        "duplicates_or_near_duplicates": duplicate_count,
    }

    log(f"nearest-neighbor stats: {result['nearest_neighbor']}")
    log(f"duplicates / distances <= 1e-12: {duplicate_count}")

    bbox = bbox_stats(points)

    if bbox["diag"] and bbox["diag"] > 0:
        log(
            "normalized nearest-neighbor stats: "
            f"mean/median/p95="
            f"{np.mean(nearest) / bbox['diag']:.9e}/"
            f"{np.percentile(nearest, 50) / bbox['diag']:.9e}/"
            f"{np.percentile(nearest, 95) / bbox['diag']:.9e}"
        )

    return result


def voxel_occupancy_statistics(points, label, voxel_sizes=None):
    points = np.asarray(points, dtype=np.float64)
    points = points[finite_rows(points)]

    log(f"\n=== {label}: VOXEL OCCUPANCY ===")

    if len(points) == 0:
        warn("Aucun point")
        return {}

    bbox = bbox_stats(points)
    minimum = bbox["min"]
    diagonal = bbox["diag"]

    if voxel_sizes is None:
        voxel_sizes = [
            diagonal / 100.0,
            diagonal / 200.0,
            diagonal / 500.0,
            diagonal / 1000.0,
        ]

    result = {}

    for voxel_size in voxel_sizes:
        if voxel_size <= 0:
            continue

        coordinates = np.floor((points - minimum) / voxel_size).astype(np.int64)
        occupied = np.unique(coordinates, axis=0)

        key = f"{voxel_size:.12g}"

        result[key] = {
            "voxel_size": float(voxel_size),
            "occupied_voxels": int(len(occupied)),
            "points": int(len(points)),
            "points_per_occupied_voxel": float(len(points) / max(1, len(occupied))),
        }

        log(
            f"voxel={voxel_size:.6f}: "
            f"occupied={len(occupied)}, "
            f"points/occupied={len(points) / max(1, len(occupied)):.3f}"
        )

    return result


# =============================================================================
# OUTLIERS
# =============================================================================

def robust_outlier_statistics(points, label):
    points = np.asarray(points, dtype=np.float64)
    points = points[finite_rows(points)]

    log(f"\n=== {label}: OUTLIERS ROBUSTES ===")

    if len(points) == 0:
        warn("Aucun point")
        return {}

    median = np.median(points, axis=0)
    distances = np.linalg.norm(points - median, axis=1)

    median_distance = np.median(distances)
    mad = np.median(np.abs(distances - median_distance))

    if mad < 1e-15:
        robust_z = np.zeros_like(distances)
    else:
        robust_z = 0.6745 * (distances - median_distance) / mad

    thresholds = [3.5, 5.0, 10.0, 20.0]
    result = {
        "center_median": median,
        "distance_stats": stats_1d(distances),
        "mad": float(mad),
        "thresholds": {},
    }

    log(f"median center: {median}")
    log(f"distance stats: {stats_1d(distances)}")
    log(f"MAD: {mad:.9f}")

    for threshold in thresholds:
        count = int(np.count_nonzero(np.abs(robust_z) > threshold))
        result["thresholds"][str(threshold)] = count
        log(f"robust outliers |z|>{threshold}: {count}")

    return result


# =============================================================================
# PLY READER
# =============================================================================

PLY_SCALAR_TYPES = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def parse_ply_header(file_object):
    format_name = None
    elements = []
    current_element = None

    while True:
        line = file_object.readline()

        if not line:
            raise RuntimeError("Header PLY tronqué")

        text = line.decode("ascii", errors="replace").strip()

        if text.startswith("format "):
            format_name = text.split()[1]

        elif text.startswith("element "):
            tokens = text.split()

            if len(tokens) != 3:
                raise RuntimeError(f"Élément PLY invalide: {text}")

            current_element = {
                "name": tokens[1],
                "count": int(tokens[2]),
                "properties": [],
            }

            elements.append(current_element)

        elif text.startswith("property "):
            if current_element is None:
                raise RuntimeError(f"Propriété PLY hors élément: {text}")

            tokens = text.split()

            if len(tokens) >= 3 and tokens[1] != "list":
                current_element["properties"].append(
                    {
                        "kind": "scalar",
                        "type": tokens[1],
                        "name": tokens[2],
                    }
                )

            elif len(tokens) >= 5 and tokens[1] == "list":
                current_element["properties"].append(
                    {
                        "kind": "list",
                        "count_type": tokens[2],
                        "item_type": tokens[3],
                        "name": tokens[4],
                    }
                )

        elif text == "end_header":
            break

    if format_name is None:
        raise RuntimeError("Format PLY manquant")

    return format_name, elements


def load_ply_points(ply_path):
    """
    Lecture robuste des coordonnées x/y/z d'un PLY ASCII ou binaire.

    La fonction supporte :
    - ASCII
    - binary_little_endian
    - binary_big_endian
    - propriétés vertex supplémentaires
    - propriétés x/y/z dans un ordre quelconque

    Les éléments après vertex ne sont pas nécessaires pour extraire les points.
    """

    ply_path = Path(ply_path)

    if not ply_path.exists():
        return np.zeros((0, 3), dtype=np.float64), {
            "exists": False,
            "format": None,
            "n_vertices": 0,
            "vertex_properties": [],
        }

    with open(ply_path, "rb") as f:
        format_name, elements = parse_ply_header(f)

        vertex_element = None

        for element in elements:
            if element["name"] == "vertex":
                vertex_element = element
                break

        if vertex_element is None:
            raise RuntimeError(f"{ply_path}: element vertex absent")

        vertex_properties = vertex_element["properties"]
        n_vertices = vertex_element["count"]

        property_names = [
            prop["name"]
            for prop in vertex_properties
            if prop["kind"] == "scalar"
        ]

        if not all(name in property_names for name in ("x", "y", "z")):
            raise RuntimeError(
                f"{ply_path}: propriétés x/y/z absentes; "
                f"propriétés={property_names}"
            )

        metadata = {
            "exists": True,
            "format": format_name,
            "n_vertices": n_vertices,
            "vertex_properties": vertex_properties,
        }

        if format_name == "ascii":
            points = []

            for _ in range(n_vertices):
                line = f.readline()

                if not line:
                    break

                tokens = line.decode("ascii", errors="replace").split()

                values = []
                token_index = 0

                for prop in vertex_properties:
                    if prop["kind"] == "scalar":
                        if token_index >= len(tokens):
                            break

                        values.append(float(tokens[token_index]))
                        token_index += 1

                    else:
                        if token_index + 1 >= len(tokens):
                            break

                        count = int(tokens[token_index])
                        token_index += 1 + count

                if len(values) != len(vertex_properties):
                    continue

                scalar_by_name = {
                    prop["name"]: values[i]
                    for i, prop in enumerate(vertex_properties)
                    if prop["kind"] == "scalar"
                }

                points.append(
                    [
                        scalar_by_name["x"],
                        scalar_by_name["y"],
                        scalar_by_name["z"],
                    ]
                )

            points = (
                np.asarray(points, dtype=np.float64)
                if points
                else np.zeros((0, 3), dtype=np.float64)
            )

            return points, metadata

        if format_name not in ("binary_little_endian", "binary_big_endian"):
            raise RuntimeError(f"Format PLY non supporté: {format_name}")

        endian = "<" if format_name == "binary_little_endian" else ">"

        struct_format = []
        property_indices = {}

        scalar_index = 0

        for prop in vertex_properties:
            if prop["kind"] != "scalar":
                raise RuntimeError(
                    "Les propriétés list dans element vertex "
                    "ne sont pas supportées par ce lecteur"
                )

            prop_type = prop["type"]

            if prop_type not in PLY_SCALAR_TYPES:
                raise RuntimeError(
                    f"Type PLY non supporté: {prop_type}"
                )

            struct_format.append(PLY_SCALAR_TYPES[prop_type])
            property_indices[prop["name"]] = scalar_index
            scalar_index += 1

        vertex_struct = struct.Struct(endian + "".join(struct_format))
        points = []

        for _ in range(n_vertices):
            raw = f.read(vertex_struct.size)

            if len(raw) != vertex_struct.size:
                break

            values = vertex_struct.unpack(raw)

            points.append(
                [
                    values[property_indices["x"]],
                    values[property_indices["y"]],
                    values[property_indices["z"]],
                ]
            )

        points = (
            np.asarray(points, dtype=np.float64)
            if points
            else np.zeros((0, 3), dtype=np.float64)
        )

        return points, metadata


def summarize_ply(label, path):
    log(f"\n=== {label}: PLY ===")
    log(f"path: {path}")

    points, metadata = load_ply_points(path)

    if not metadata["exists"]:
        warn("PLY absent")
        return {
            "metadata": metadata,
            "bbox": bbox_stats(points),
        }

    log(f"format: {metadata['format']}")
    log(f"n_vertices_header: {metadata['n_vertices']}")
    log(f"n_vertices_loaded: {len(points)}")
    log(f"vertex_properties: {metadata['vertex_properties']}")

    if len(points) != metadata["n_vertices"]:
        warn(
            f"Nombre de points lu différent du header: "
            f"{len(points)} != {metadata['n_vertices']}"
        )

    stats = print_bbox_stats("PLY points", points)

    invalid = ~finite_rows(points)

    if np.any(invalid):
        indices = np.flatnonzero(invalid)
        warn(
            f"{len(indices)} points PLY avec NaN/Inf; "
            f"premiers indices={indices[:20].tolist()}"
        )

    return {
        "metadata": metadata,
        "bbox": stats,
        "points": points,
    }


# =============================================================================
# COMPARAISONS
# =============================================================================

def compare_pose_sets(label_a, centers_a, label_b, centers_b):
    log("\n=== COMPARAISON DES POSES ===")

    if len(centers_a) == 0 or len(centers_b) == 0:
        warn("Impossible de comparer: pose vide")
        return {}

    bbox_a = bbox_stats(centers_a)
    bbox_b = bbox_stats(centers_b)

    pairwise_a = pairwise_baselines(centers_a)
    pairwise_b = pairwise_baselines(centers_b)

    consecutive_a = consecutive_baselines(centers_a)
    consecutive_b = consecutive_baselines(centers_b)

    def ratio(a, b):
        if a is None or b is None or abs(b) < 1e-15:
            return np.nan
        return a / b

    log(
        f"frames ratio {label_a}/{label_b}: "
        f"{len(centers_a)}/{len(centers_b)} = "
        f"{len(centers_a) / max(1, len(centers_b)):.9f}"
    )

    log(
        f"bbox diag ratio {label_a}/{label_b}: "
        f"{ratio(bbox_a['diag'], bbox_b['diag']):.9f}"
    )

    if len(pairwise_a) and len(pairwise_b):
        log(
            f"pairwise mean ratio {label_a}/{label_b}: "
            f"{ratio(np.mean(pairwise_a), np.mean(pairwise_b)):.9f}"
        )
        log(
            f"pairwise p50 ratio {label_a}/{label_b}: "
            f"{ratio(np.percentile(pairwise_a, 50), np.percentile(pairwise_b, 50)):.9f}"
        )

    if len(consecutive_a) and len(consecutive_b):
        log(
            f"consecutive mean ratio {label_a}/{label_b}: "
            f"{ratio(np.mean(consecutive_a), np.mean(consecutive_b)):.9f}"
        )

    return {
        "frames_ratio": len(centers_a) / max(1, len(centers_b)),
        "bbox_diag_ratio": ratio(bbox_a["diag"], bbox_b["diag"]),
    }


def compare_points_clouds(label_a, points_a, label_b, points_b):
    log("\n=== COMPARAISON DES NUAGEs 3D ===")

    stats_a = print_bbox_stats(label_a, points_a)
    stats_b = print_bbox_stats(label_b, points_b)

    if stats_a["diag"] and stats_b["diag"]:
        log(
            f"bbox diag ratio {label_a}/{label_b}: "
            f"{stats_a['diag'] / stats_b['diag']:.9f}"
        )

    if len(points_a) and len(points_b):
        mean_a = np.mean(points_a[finite_rows(points_a)], axis=0)
        mean_b = np.mean(points_b[finite_rows(points_b)], axis=0)

        log(f"mean difference {label_a} - {label_b}: {mean_a - mean_b}")
        log(f"mean distance: {np.linalg.norm(mean_a - mean_b):.9f}")

    return {
        label_a: stats_a,
        label_b: stats_b,
    }


def compare_camera_parameters(label_a, cameras_a, label_b, cameras_b):
    log("\n=== COMPARAISON DES CAMERAS ===")

    log(f"{label_a}: {len(cameras_a)} caméra(s)")
    log(f"{label_b}: {len(cameras_b)} caméra(s)")

    for label, cameras in [(label_a, cameras_a), (label_b, cameras_b)]:
        for camera_id, camera in cameras.items():
            intrinsics = camera_intrinsics(camera)
            log(
                f"{label} camera_id={camera_id}: "
                f"model={camera['model']}, "
                f"size={camera['w']}x{camera['h']}, "
                f"fx={intrinsics['fx']:.9f}, "
                f"fy={intrinsics['fy']:.9f}, "
                f"cx={intrinsics['cx']:.9f}, "
                f"cy={intrinsics['cy']:.9f}"
            )

    return {label_a: cameras_a, label_b: cameras_b}


def compare_transforms_to_colmap(
    label,
    centers_json,
    rotations_json,
    names_json,
    colmap_images,
):
    log(f"\n=== {label}: JSON VS COLMAP ===")

    common = [
        name
        for name in names_json
        if name in colmap_images
    ]

    if not common:
        warn("Aucune image commune")
        return {}

    json_index = {
        name: index
        for index, name in enumerate(names_json)
    }

    center_errors = []
    rotation_errors = []

    for name in common:
        index = json_index[name]
        colmap = colmap_images[name]

        center_errors.append(
            np.linalg.norm(centers_json[index] - colmap["C"])
        )

        rotation_errors.append(
            np.linalg.norm(rotations_json[index] - colmap["R"])
        )

    center_errors = np.asarray(center_errors)
    rotation_errors = np.asarray(rotation_errors)

    log(f"common frames: {len(common)}")
    log(f"center error stats: {stats_1d(center_errors)}")
    log(f"rotation error stats: {stats_1d(rotation_errors)}")

    return {
        "common_frames": len(common),
        "center_error": stats_1d(center_errors),
        "rotation_error": stats_1d(rotation_errors),
    }


# =============================================================================
# DATASET ANALYSIS
# =============================================================================

def analyze_dataset(name, paths, args):
    log("\n" + "=" * 90)
    log(f"ANALYSE COMPLETE DU DATASET: {name}")
    log("=" * 90)

    result = {
        "name": name,
        "paths": {},
    }

    if not paths["root"].exists():
        error(f"Dossier dataset absent: {paths['root']}")
        return result

    log(f"root: {paths['root']}")

    # -------------------------------------------------------------------------
    # Files
    # -------------------------------------------------------------------------

    log("\n--- FICHIERS ---")

    file_keys = [
        "transforms",
        "normalization",
        "georeferencing",
        "sparse_pc",
        "model_ply",
        "colmap",
        "colmap_txt",
        "images",
    ]

    for key in file_keys:
        path = paths[key]
        result["paths"][key] = str(path)

        if path.is_dir():
            log(f"{key}: directory exists={path.exists()} path={path}")
        else:
            print_file_info(key, path, with_hash=args.hash_files)

    # -------------------------------------------------------------------------
    # Transforms
    # -------------------------------------------------------------------------

    centers_json, rotations_json, names_json, transforms = load_transforms(
        paths["transforms"]
    )

    result["transforms"] = {
        "n_frames": len(centers_json),
        "metadata": transforms,
    }

    pose_summary = summarize_pose_set(
        name,
        centers_json,
        rotations_json,
        transforms,
    )

    result["pose_summary"] = pose_summary

    # -------------------------------------------------------------------------
    # COLMAP TXT
    # -------------------------------------------------------------------------

    txt_dir = find_colmap_txt_dir(paths)
    result["colmap_txt_dir"] = str(txt_dir) if txt_dir else None

    if txt_dir is None:
        warn(
            f"{name}: aucun modèle TXT trouvé. "
            f"Attendu sous {paths['colmap_txt']} ou dans les dossiers auxiliaires."
        )

        colmap_images = {}
        cameras = {}
        points_data = None

    else:
        log(f"\nCOLMAP TXT utilisé: {txt_dir}")

        cameras_path = txt_dir / "cameras.txt"
        images_path = txt_dir / "images.txt"
        points_path = txt_dir / "points3D.txt"

        cameras = load_colmap_cameras_txt(cameras_path)
        colmap_images = load_colmap_images_txt(images_path)
        points_data = load_colmap_points3d_txt(points_path)

        summarize_cameras(name, cameras)
        summarize_image_tracks(name, colmap_images)
        summarize_points3d(name, points_data)

        consistency = track_consistency(points_data, colmap_images)

        log(f"\n=== {name}: TRACK CONSISTENCY ===")
        log(f"total tracks: {consistency['total_tracks']}")
        log(f"tracks with missing image: {consistency['missing_images']}")
        log(f"tracks with invalid point2D index: {consistency['bad_point2d_indices']}")

        result["track_consistency"] = consistency

        reprojection = compute_reprojection_statistics(
            points_data,
            colmap_images,
            cameras,
        )

        print_reprojection_statistics(name, reprojection)
        result["reprojection"] = reprojection

        compare_transforms_to_colmap(
            name,
            centers_json,
            rotations_json,
            names_json,
            colmap_images,
        )

    result["colmap_images"] = colmap_images
    result["cameras"] = cameras
    result["points_data"] = points_data

    # -------------------------------------------------------------------------
    # Points 3D density
    # -------------------------------------------------------------------------

    if points_data is not None:
        points_xyz = points_data["xyz"]

        nearest_stats = nearest_neighbor_statistics(
            points_xyz,
            f"{name} COLMAP points3D.txt",
        )

        voxel_stats = voxel_occupancy_statistics(
            points_xyz,
            f"{name} COLMAP points3D.txt",
        )

        outlier_stats = robust_outlier_statistics(
            points_xyz,
            f"{name} COLMAP points3D.txt",
        )

        result["nearest_neighbors_colmap"] = nearest_stats
        result["voxel_colmap"] = voxel_stats
        result["outliers_colmap"] = outlier_stats

    # -------------------------------------------------------------------------
    # sparse_pc.ply
    # -------------------------------------------------------------------------

    ply_result = summarize_ply(
        f"{name} sparse_pc.ply",
        paths["sparse_pc"],
    )

    result["sparse_pc"] = ply_result

    if "points" in ply_result:
        ply_points = ply_result["points"]

        result["nearest_neighbors_ply"] = nearest_neighbor_statistics(
            ply_points,
            f"{name} sparse_pc.ply",
        )

        result["voxel_ply"] = voxel_occupancy_statistics(
            ply_points,
            f"{name} sparse_pc.ply",
        )

        result["outliers_ply"] = robust_outlier_statistics(
            ply_points,
            f"{name} sparse_pc.ply",
        )

    if args.print_ply_header:
        print_ply_header(paths["sparse_pc"])

    return result


# =============================================================================
# NORMALISATION / CROSS DATASET
# =============================================================================

def print_normalization(path, label):
    if not path.exists():
        warn(f"{label}: sidecar absent: {path}")
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warn(f"{label}: sidecar illisible: {exc}")
        return None

    log(f"\n=== {label}: NORMALISATION ===")
    log(json.dumps(data, indent=2, ensure_ascii=False))

    return data


def compare_normalizations(label_a, norm_a, label_b, norm_b):
    log("\n=== COMPARAISON DES NORMALISATIONS ===")

    if norm_a is None or norm_b is None:
        warn("Sidecar manquant pour l'une des deux datasets")
        return

    for key in [
        "uniform_scale_applied",
        "camera_radius_before",
        "camera_radius_after",
    ]:
        if key in norm_a or key in norm_b:
            log(
                f"{key}: "
                f"{label_a}={norm_a.get(key)} ; "
                f"{label_b}={norm_b.get(key)}"
            )

    a_translation = norm_a.get("translation_subtracted_world_xyz")
    b_translation = norm_b.get("translation_subtracted_world_xyz")

    if a_translation is not None and b_translation is not None:
        a_translation = np.asarray(a_translation, dtype=np.float64)
        b_translation = np.asarray(b_translation, dtype=np.float64)

        log(
            "translation difference "
            f"{label_a} - {label_b}: "
            f"{a_translation - b_translation}"
        )


def compare_initialization_variables(result_a, result_b, label_a, label_b):
    log("\n=== COMPARAISON DES VARIABLES CRITIQUES POUR SPLATFACTO ===")

    def get(result, *keys):
        value = result

        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)

        return value

    values = [
        (
            "COLMAP point count",
            get(result_a, "points_data", "xyz"),
            get(result_b, "points_data", "xyz"),
        ),
        (
            "PLY point count",
            get(result_a, "sparse_pc", "points"),
            get(result_b, "sparse_pc", "points"),
        ),
        (
            "COLMAP bbox diag",
            get(result_a, "points_data", "xyz"),
            get(result_b, "points_data", "xyz"),
        ),
        (
            "PLY bbox diag",
            get(result_a, "sparse_pc", "bbox", "diag"),
            get(result_b, "sparse_pc", "bbox", "diag"),
        ),
    ]

    for label, a, b in values:
        if isinstance(a, np.ndarray):
            a = len(a)

        if isinstance(b, np.ndarray):
            b = len(b)

        log(f"{label}: {label_a}={a} ; {label_b}={b}")


def print_direct_comparison(result_a, result_b, label_a, label_b):
    log("\n" + "=" * 90)
    log(f"COMPARAISON DIRECTE: {label_a} VS {label_b}")
    log("=" * 90)

    def cloud_from_result(result, source):
        if source == "colmap":
            points_data = result.get("points_data")
            return (
                points_data["xyz"]
                if points_data is not None
                else np.zeros((0, 3))
            )

        if source == "ply":
            ply = result.get("sparse_pc", {})
            return ply.get("points", np.zeros((0, 3)))

        return np.zeros((0, 3))

    for source in ["colmap", "ply"]:
        points_a = cloud_from_result(result_a, source)
        points_b = cloud_from_result(result_b, source)

        compare_points_clouds(
            f"{label_a} {source}",
            points_a,
            f"{label_b} {source}",
            points_b,
        )

        nearest_neighbor_statistics(
            points_a,
            f"{label_a} {source}",
        )

        nearest_neighbor_statistics(
            points_b,
            f"{label_b} {source}",
        )

    compare_initialization_variables(
        result_a,
        result_b,
        label_a,
        label_b,
    )


# =============================================================================
# JSON SERIALIZATION
# =============================================================================

def make_json_safe(value):
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        if value.dtype.kind in "f":
            return value.tolist()
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None
        return float(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
            if key != "points"
            and key != "tracks"
            and key != "colmap_images"
            and key != "cameras"
            and key != "points_data"
        }

    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    return value


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Comparaison exhaustive de deux datasets COLMAP/Nerfstudio "
            "pour diagnostiquer les différences d'initialisation Splatfacto."
        )
    )

    parser.add_argument(
        "--dataset1",
        required=True,
        help="Nom du premier dataset, sous BASE_DIR.",
    )

    parser.add_argument(
        "--dataset2",
        required=True,
        help="Nom du deuxième dataset, sous BASE_DIR.",
    )

    parser.add_argument(
        "--base-dir",
        default=None,
        help="Répertoire contenant les deux datasets.",
    )

    parser.add_argument(
        "--compare-json-to-colmap",
        action="store_true",
        help="Conserver la comparaison transforms.json vs images.txt.",
    )

    parser.add_argument(
        "--hash-files",
        action="store_true",
        help="Calculer les SHA256 des fichiers principaux.",
    )

    parser.add_argument(
        "--print-ply-header",
        action="store_true",
        help="Afficher le header complet des PLY.",
    )

    parser.add_argument(
        "--report-json",
        default=None,
        help="Écrire les résultats sérialisables dans un JSON.",
    )

    args = parser.parse_args()

    global BASE_DIR

    if args.base_dir is not None:
        BASE_DIR = Path(args.base_dir).resolve()

    paths_a = dataset_paths(args.dataset1)
    paths_b = dataset_paths(args.dataset2)

    result_a = analyze_dataset(
        args.dataset1,
        paths_a,
        args,
    )

    result_b = analyze_dataset(
        args.dataset2,
        paths_b,
        args,
    )

    # -------------------------------------------------------------------------
    # Poses
    # -------------------------------------------------------------------------

    centers_a, rotations_a, names_a, metadata_a = load_transforms(
        paths_a["transforms"]
    )

    centers_b, rotations_b, names_b, metadata_b = load_transforms(
        paths_b["transforms"]
    )

    compare_pose_sets(
        args.dataset1,
        centers_a,
        args.dataset2,
        centers_b,
    )

    # -------------------------------------------------------------------------
    # Comparaison fine transforms.json par nom d'image
    # -------------------------------------------------------------------------

    compare_transforms_by_image_name(
        args.dataset1,
        paths_a["transforms"],
        args.dataset2,
        paths_b["transforms"],
    )

    detect_pose_permutation_by_nearest_center(
        args.dataset1,
        paths_a["transforms"],
        args.dataset2,
        paths_b["transforms"],
    )

    # -------------------------------------------------------------------------
    # Projection du sparse_pc.ply via transforms.json
    # -------------------------------------------------------------------------

    visibility_a = sparse_pc_visibility_from_transforms(
        args.dataset1,
        paths_a["transforms"],
        paths_a["sparse_pc"],
        max_points=200000,
    )

    visibility_b = sparse_pc_visibility_from_transforms(
        args.dataset2,
        paths_b["transforms"],
        paths_b["sparse_pc"],
        max_points=200000,
    )

    compare_sparse_pc_visibility(
        args.dataset1,
        visibility_a,
        args.dataset2,
        visibility_b,
    )

    # -------------------------------------------------------------------------
    # Cameras
    # -------------------------------------------------------------------------

    cameras_a = result_a.get("cameras", {})
    cameras_b = result_b.get("cameras", {})

    compare_camera_parameters(
        args.dataset1,
        cameras_a,
        args.dataset2,
        cameras_b,
    )

    # -------------------------------------------------------------------------
    # JSON -> COLMAP option
    # -------------------------------------------------------------------------

    if args.compare_json_to_colmap:
        images_a = result_a.get("colmap_images", {})
        images_b = result_b.get("colmap_images", {})

        if images_a:
            compare_transforms_to_colmap(
                args.dataset1,
                centers_a,
                rotations_a,
                names_a,
                images_a,
            )

        if images_b:
            compare_transforms_to_colmap(
                args.dataset2,
                centers_b,
                rotations_b,
                names_b,
                images_b,
            )

    # -------------------------------------------------------------------------
    # Sidecars
    # -------------------------------------------------------------------------

    norm_a = print_normalization(
        paths_a["normalization"],
        f"{args.dataset1} transforms_normalization.json",
    )

    norm_b = print_normalization(
        paths_b["normalization"],
        f"{args.dataset2} transforms_normalization.json",
    )

    compare_normalizations(
        args.dataset1,
        norm_a,
        args.dataset2,
        norm_b,
    )

    # -------------------------------------------------------------------------
    # Comparaison points / densité / initialisation
    # -------------------------------------------------------------------------

    print_direct_comparison(
        result_a,
        result_b,
        args.dataset1,
        args.dataset2,
    )

    # -------------------------------------------------------------------------
    # Résumé final de diagnostic
    # -------------------------------------------------------------------------

    log("\n" + "=" * 90)
    log("DIAGNOSTIC FINAL AUTOMATIQUE")
    log("=" * 90)

    for label, result in [
        (args.dataset1, result_a),
        (args.dataset2, result_b),
    ]:
        points_data = result.get("points_data")
        ply = result.get("sparse_pc", {})

        if points_data is not None:
            xyz = points_data["xyz"]
            finite_count = int(np.count_nonzero(finite_rows(xyz)))

            log(
                f"{label}: COLMAP points3D={len(xyz)}, "
                f"finite={finite_count}"
            )

        ply_points = ply.get("points")

        if ply_points is not None:
            finite_count = int(np.count_nonzero(finite_rows(ply_points)))

            log(
                f"{label}: sparse_pc.ply={len(ply_points)}, "
                f"finite={finite_count}"
            )

    log("\nPoints à comparer en priorité pour le split Splatfacto:")
    log("  1. nearest-neighbor median/p95")
    log("  2. reprojection error median/p95")
    log("  3. profondeur camera median/p95")
    log("  4. track length median/p95")
    log("  5. bbox diag COLMAP vs bbox diag PLY")
    log("  6. nombre de points COLMAP vs nombre initial de GS")
    log("  7. sidecar uniform_scale_applied")
    log("  8. présence de doublons ou points concentrés dans quelques voxels")

    # -------------------------------------------------------------------------
    # Rapport JSON
    # -------------------------------------------------------------------------

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "base_dir": str(BASE_DIR),
            "dataset1": args.dataset1,
            "dataset2": args.dataset2,
            "dataset1_result": make_json_safe(result_a),
            "dataset2_result": make_json_safe(result_b),
        }

        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        log(f"\n[OK] Rapport JSON écrit: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        error("Interruption utilisateur")
        sys.exit(130)
    except Exception as exc:
        error(str(exc))
        raise

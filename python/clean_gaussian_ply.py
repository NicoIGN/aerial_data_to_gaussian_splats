#!/usr/bin/env python3

import argparse
from pathlib import Path
from collections import deque
import sys
import traceback

import numpy as np
from plyfile import PlyData, PlyElement


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_opacity(raw_opacity):
    return sigmoid(raw_opacity)


def decode_scales(vertices):
    names = vertices.dtype.names

    if all(n in names for n in ("scale_0", "scale_1", "scale_2")):
        scales_log = np.column_stack([
            vertices["scale_0"],
            vertices["scale_1"],
            vertices["scale_2"]
        ])
        return np.exp(scales_log)

    if all(n in names for n in ("sx", "sy", "sz")):
        return np.column_stack([
            np.abs(vertices["sx"]),
            np.abs(vertices["sy"]),
            np.abs(vertices["sz"]),
        ])

    return None


def compute_scale_metric(scales, metric):
    if metric == "max":
        return np.max(scales, axis=1)
    elif metric == "mean":
        return np.mean(scales, axis=1)
    elif metric == "norm":
        return np.linalg.norm(scales, axis=1)
    raise ValueError(metric)


def get_xyz(vertices):
    return np.column_stack([
        vertices["x"],
        vertices["y"],
        vertices["z"]
    ])


def robust_center(xyz):
    return np.median(xyz, axis=0)


def robust_sigma(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad + 1e-12


def print_stage(title):
    print(f"\n[{title}]")


def estimate_scene_plane_normal(xyz):
    if len(xyz) == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), {}

    center = np.median(xyz, axis=0)
    centered = xyz - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    normal = eigvecs[:, 0]
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    info = {
        "plane_center": center,
        "eigenvalues": eigvals,
        "normal": normal,
    }
    return normal, info


def split_plane_normal_components(points, center, plane_normal=None):
    diff = points - center

    if plane_normal is None:
        plane_dist = np.linalg.norm(diff, axis=-1)
        normal_dist = np.zeros_like(plane_dist)
        return plane_dist, normal_dist

    n = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
    signed_normal = np.sum(diff * n, axis=-1)
    normal_vec = signed_normal[..., None] * n
    planar_vec = diff - normal_vec

    plane_dist = np.linalg.norm(planar_vec, axis=-1)
    normal_dist = np.abs(signed_normal)
    return plane_dist, normal_dist


def get_internal_params(scene_type, border_strictness):
    s = float(np.clip(border_strictness, 0.0, 1.0))
    s_eff = 0.7 * s + 0.3 * (s ** 2)

    if scene_type == "compact":
        base = {
            "center_scale_percentile": 75.0,
            "center_opacity_percentile": 75.0,
            "center_keep_percentile": 82.0,
            "voxel_min_points": 3,
            "voxel_center_sigma": 2.4,
            "voxel_normal_sigma": 6.0,
            "voxel_max_fraction": 0.40,
            "attach_layers": 1,
            "attach_min_points": 2,
            "voxel_scale_percentile": 95.0,
        }

        radial_density_bias = 0.3 + 4.0 * s_eff
        scale_bias = 0.3 + 5.0 * s_eff
        attach_radial_density_bias = 0.2 + 2.5 * s_eff

        voxel_center_sigma = max(1.8, base["voxel_center_sigma"] - 0.9 * s_eff)
        voxel_scale_percentile = max(86.0, base["voxel_scale_percentile"] - 7.0 * s_eff)
        voxel_max_fraction = max(0.22, base["voxel_max_fraction"] - 0.30 * s_eff)

        if s_eff >= 0.9:
            attach_layers = 0
        elif s_eff >= 0.65:
            attach_layers = max(0, base["attach_layers"] - 1)
        else:
            attach_layers = base["attach_layers"]

        attach_min_points = base["attach_min_points"] + int(s_eff >= 0.75)
        voxel_min_points = base["voxel_min_points"] + int(s_eff >= 0.9)

    elif scene_type == "wide":
        base = {
            "center_scale_percentile": 99.0,
            "center_opacity_percentile": 45.0,
            "center_keep_percentile": 99.0,
            "voxel_min_points": 1,
            "voxel_center_sigma": 4.5,
            "voxel_normal_sigma": 12.0,
            "voxel_max_fraction": 1.00,
            "attach_layers": 5,
            "attach_min_points": 1,
            "voxel_scale_percentile": 99.8,
        }

        # Very permissive for broad textured surfaces like tables
        radial_density_bias = 0.0 + 0.5 * s_eff
        scale_bias = 0.0 + 0.3 * s_eff
        attach_radial_density_bias = 0.0 + 0.3 * s_eff

        voxel_center_sigma = max(4.0, base["voxel_center_sigma"] - 0.2 * s_eff)
        voxel_scale_percentile = max(99.0, base["voxel_scale_percentile"] - 0.5 * s_eff)
        voxel_max_fraction = 1.00

        # Do not aggressively reduce attachment in wide mode
        attach_layers = base["attach_layers"]
        attach_min_points = 1
        voxel_min_points = 1

    else:  # balanced
        base = {
            "center_scale_percentile": 85.0,
            "center_opacity_percentile": 70.0,
            "center_keep_percentile": 88.0,
            "voxel_min_points": 2,
            "voxel_center_sigma": 3.0,
            "voxel_normal_sigma": 7.0,
            "voxel_max_fraction": 0.70,
            "attach_layers": 2,
            "attach_min_points": 1,
            "voxel_scale_percentile": 96.0,
        }

        radial_density_bias = 0.3 + 4.0 * s_eff
        scale_bias = 0.3 + 5.0 * s_eff
        attach_radial_density_bias = 0.2 + 2.5 * s_eff

        voxel_center_sigma = max(1.8, base["voxel_center_sigma"] - 0.9 * s_eff)
        voxel_scale_percentile = max(86.0, base["voxel_scale_percentile"] - 7.0 * s_eff)
        voxel_max_fraction = max(0.22, base["voxel_max_fraction"] - 0.30 * s_eff)

        if s_eff >= 0.9:
            attach_layers = 0
        elif s_eff >= 0.65:
            attach_layers = max(0, base["attach_layers"] - 1)
        else:
            attach_layers = base["attach_layers"]

        attach_min_points = base["attach_min_points"] + int(s_eff >= 0.75)
        voxel_min_points = base["voxel_min_points"] + int(s_eff >= 0.9)

    return {
        "center_scale_percentile": base["center_scale_percentile"],
        "center_opacity_percentile": base["center_opacity_percentile"],
        "center_keep_percentile": base["center_keep_percentile"],
        "voxel_min_points": voxel_min_points,
        "voxel_center_sigma": voxel_center_sigma,
        "voxel_normal_sigma": base["voxel_normal_sigma"],
        "voxel_max_fraction": voxel_max_fraction,
        "attach_layers": attach_layers,
        "attach_min_points": attach_min_points,
        "voxel_scale_percentile": voxel_scale_percentile,
        "voxel_radial_density_bias": radial_density_bias,
        "voxel_scale_bias": scale_bias,
        "attach_radial_density_bias": attach_radial_density_bias,
    }
    

def validate_args(args):
    if not (0.0 <= args.min_opacity <= 1.0):
        raise ValueError("--min-opacity must be in [0, 1]")
    if not (0.0 < args.max_scale_percentile <= 100.0):
        raise ValueError("--max-scale-percentile must be in (0, 100]")
    if not (0.0 < args.center_percentile <= 100.0):
        raise ValueError("--center-percentile must be in (0, 100]")
    if args.voxel_grid_size < 8:
        raise ValueError("--voxel-grid-size must be >= 8")
    if not (0.0 <= args.border_strictness <= 1.0):
        raise ValueError("--border-strictness must be in [0, 1]")


def opacity_filter(vertices, min_opacity):
    if "opacity" not in vertices.dtype.names:
        return np.ones(len(vertices), dtype=bool)
    opacity = decode_opacity(vertices["opacity"])
    return opacity >= min_opacity


def global_scale_filter(vertices, percentile, metric):
    scales = decode_scales(vertices)
    if scales is None:
        return np.ones(len(vertices), dtype=bool), None

    scale_metric = compute_scale_metric(scales, metric)
    threshold = np.percentile(scale_metric, percentile)
    keep = scale_metric <= threshold
    return keep, threshold


def estimate_scene_center(vertices, scale_metric_name, params, distance_mode, plane_normal):
    xyz = get_xyz(vertices)
    n = len(xyz)

    if n == 0:
        return np.zeros(3, dtype=np.float64), {}

    base_center = robust_center(xyz)

    if distance_mode == "planar-aware" and plane_normal is not None:
        base_dist, _ = split_plane_normal_components(xyz, base_center, plane_normal)
    else:
        base_dist = np.linalg.norm(xyz - base_center, axis=1)

    if "opacity" in vertices.dtype.names:
        opacity = decode_opacity(vertices["opacity"])
    else:
        opacity = np.ones(n, dtype=np.float64)

    scales = decode_scales(vertices)
    if scales is not None:
        scale_metric = compute_scale_metric(scales, scale_metric_name)
    else:
        scale_metric = np.ones(n, dtype=np.float64)

    scale_thr = np.percentile(scale_metric, params["center_scale_percentile"])
    opacity_thr = np.percentile(opacity, params["center_opacity_percentile"])
    dist_thr = np.percentile(base_dist, params["center_keep_percentile"])

    keep = (
        (scale_metric <= scale_thr) &
        (opacity >= opacity_thr) &
        (base_dist <= dist_thr)
    )

    if np.sum(keep) < max(128, int(0.001 * n)):
        keep = base_dist <= np.percentile(base_dist, 85.0)

    if np.sum(keep) == 0:
        center = base_center
    else:
        center = np.median(xyz[keep], axis=0)

    info = {
        "selected_points": int(np.sum(keep)),
        "scale_threshold": float(scale_thr),
        "opacity_threshold": float(opacity_thr),
        "distance_threshold": float(dist_thr),
        "center": center,
    }
    return center, info


def center_filter(vertices, percentile, scene_center, distance_mode, plane_normal):
    xyz = get_xyz(vertices)

    if distance_mode == "planar-aware" and plane_normal is not None:
        plane_dist, _ = split_plane_normal_components(xyz, scene_center, plane_normal)
        dist = plane_dist
    else:
        dist = np.linalg.norm(xyz - scene_center, axis=1)

    threshold = np.percentile(dist, percentile)
    keep = dist <= threshold
    return keep, threshold


def compute_voxel_indices(xyz, mins, voxel_size, grid_size):
    rel = (xyz - mins) / voxel_size
    idx = np.floor(rel).astype(np.int32)
    idx = np.clip(idx, 0, grid_size - 1)
    return idx


def make_neighbor_offsets():
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offsets.append((dx, dy, dz))
    return offsets


def voxel_cluster_filter(vertices, scene_center, voxel_grid_size, params, distance_mode, plane_normal):
    xyz = get_xyz(vertices)
    n = len(xyz)

    if n == 0:
        return np.zeros(0, dtype=bool), {}

    if distance_mode == "planar-aware" and plane_normal is not None:
        point_plane_dist, point_normal_dist = split_plane_normal_components(xyz, scene_center, plane_normal)
        dist_ref = point_plane_dist
        normal_med = np.median(point_normal_dist)
        normal_sig = robust_sigma(point_normal_dist)
        normal_limit = normal_med + params["voxel_normal_sigma"] * normal_sig
    else:
        dist_ref = np.linalg.norm(xyz - scene_center, axis=1)
        point_normal_dist = np.zeros_like(dist_ref)
        normal_limit = np.inf

    dist_med = np.median(dist_ref)
    dist_sig = robust_sigma(dist_ref)
    radial_limit = dist_med + params["voxel_center_sigma"] * dist_sig

    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    extent = np.maximum(maxs - mins, 1e-9)
    voxel_size = extent / float(voxel_grid_size)

    voxel_idx = compute_voxel_indices(xyz, mins, voxel_size, voxel_grid_size)

    voxel_keys, inverse, counts = np.unique(
        voxel_idx, axis=0, return_inverse=True, return_counts=True
    )

    voxel_centers = mins + (voxel_keys.astype(np.float64) + 0.5) * voxel_size

    if distance_mode == "planar-aware" and plane_normal is not None:
        voxel_plane_dist, voxel_normal_dist = split_plane_normal_components(voxel_centers, scene_center, plane_normal)
        voxel_center_dist = voxel_plane_dist
    else:
        voxel_center_dist = np.linalg.norm(voxel_centers - scene_center, axis=1)
        voxel_normal_dist = np.zeros_like(voxel_center_dist)

    voxel_radial_norm = voxel_center_dist / (radial_limit + 1e-12)

    scales = decode_scales(vertices)
    if scales is not None:
        point_scale_metric = compute_scale_metric(scales, "max")
        voxel_scale = np.zeros(len(voxel_keys), dtype=np.float64)

        order = np.argsort(inverse)
        inv_sorted = inverse[order]
        scale_sorted = point_scale_metric[order]
        starts = np.concatenate(([0], np.flatnonzero(np.diff(inv_sorted)) + 1, [len(order)]))

        for i in range(len(starts) - 1):
            a, b = starts[i], starts[i + 1]
            voxel_id = inv_sorted[a]
            voxel_scale[voxel_id] = np.median(scale_sorted[a:b])

        voxel_scale_limit = np.percentile(voxel_scale, params["voxel_scale_percentile"])
    else:
        voxel_scale = np.ones(len(voxel_keys), dtype=np.float64)
        voxel_scale_limit = np.inf

    dense_required = params["voxel_min_points"] * (
        1.0 + params["voxel_radial_density_bias"] * (voxel_radial_norm ** 2)
    )

    if np.isfinite(voxel_scale_limit):
        scale_allowed = voxel_scale_limit / (
            1.0 + params["voxel_scale_bias"] * np.maximum(voxel_radial_norm - 0.5, 0.0) ** 2
        )
        scale_ok = voxel_scale <= scale_allowed
    else:
        scale_ok = np.ones(len(voxel_keys), dtype=bool)

    radial_ok = voxel_center_dist <= radial_limit
    normal_ok = voxel_normal_dist <= normal_limit
    dense_ok = counts >= dense_required

    active_mask = dense_ok & radial_ok & normal_ok & scale_ok

    active_voxel_keys = voxel_keys[active_mask]
    active_center_dist = voxel_center_dist[active_mask]

    if len(active_voxel_keys) == 0:
        return np.zeros(n, dtype=bool), {
            "num_occupied_voxels": len(voxel_keys),
            "num_active_voxels": 0,
            "selected_voxels": 0,
            "selected_points": 0,
            "radial_limit": radial_limit,
            "normal_limit": normal_limit,
            "voxel_scale_limit": voxel_scale_limit,
            "voxel_size": voxel_size,
            "stop_reason": "no_active_voxels",
        }

    active_dict = {tuple(v.tolist()): i for i, v in enumerate(active_voxel_keys)}
    seed_idx = int(np.argmin(active_center_dist))
    seed_voxel = tuple(active_voxel_keys[seed_idx].tolist())

    offsets = make_neighbor_offsets()
    selected = {seed_voxel}
    q = deque([seed_voxel])

    max_voxels = max(1, int(np.floor(params["voxel_max_fraction"] * len(active_voxel_keys))))
    stop_reason = "frontier_exhausted"

    while q:
        if len(selected) >= max_voxels:
            stop_reason = "max_fraction_reached"
            break

        v = q.popleft()
        x, y, z = v

        for dx, dy, dz in offsets:
            nb = (x + dx, y + dy, z + dz)
            if nb in active_dict and nb not in selected:
                selected.add(nb)
                q.append(nb)
                if len(selected) >= max_voxels:
                    stop_reason = "max_fraction_reached"
                    break

        if len(selected) >= max_voxels:
            break

    if params["attach_layers"] > 0:
        occupied_dict = {tuple(v.tolist()): i for i, v in enumerate(voxel_keys)}
        frontier = set(selected)
        selected_expanded = set(selected)

        for _ in range(params["attach_layers"]):
            new_frontier = set()

            for v in frontier:
                x, y, z = v
                for dx, dy, dz in offsets:
                    nb = (x + dx, y + dy, z + dz)
                    if nb in selected_expanded or nb not in occupied_dict:
                        continue

                    occ_idx = occupied_dict[nb]
                    radial_norm = voxel_radial_norm[occ_idx]

                    required_attach = params["attach_min_points"] * (
                        1.0 + params["attach_radial_density_bias"] * (radial_norm ** 2)
                    )

                    if counts[occ_idx] < required_attach:
                        continue
                    if voxel_center_dist[occ_idx] > radial_limit:
                        continue
                    if voxel_normal_dist[occ_idx] > normal_limit:
                        continue
                    if not scale_ok[occ_idx]:
                        continue

                    selected_expanded.add(nb)
                    new_frontier.add(nb)

            if not new_frontier:
                break

            frontier = new_frontier

        selected = selected_expanded

    selected_struct = set(selected)
    keep_points = np.array([tuple(v.tolist()) in selected_struct for v in voxel_idx], dtype=bool)

    return keep_points, {
        "num_occupied_voxels": len(voxel_keys),
        "num_active_voxels": len(active_voxel_keys),
        "selected_voxels": len(selected),
        "selected_points": int(np.sum(keep_points)),
        "radial_limit": radial_limit,
        "normal_limit": normal_limit,
        "voxel_scale_limit": voxel_scale_limit,
        "voxel_size": voxel_size,
        "stop_reason": stop_reason,
        "seed_voxel": seed_voxel,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--min-opacity", type=float, default=0.005)
    p.add_argument("--max-scale-percentile", type=float, default=99.0)
    p.add_argument("--center-percentile", type=float, default=98.0)
    p.add_argument("--voxel-grid-size", type=int, default=100)
    p.add_argument("--scene-type", choices=["compact", "balanced", "wide"], default="balanced")
    p.add_argument("--border-strictness", type=float, default=0.5)
    p.add_argument("--distance-mode", choices=["euclidean", "planar-aware"], default="planar-aware")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--recenter", action="store_true")
    p.add_argument("--supersplat", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    validate_args(args)

    if args.recenter and args.supersplat:
        raise ValueError("--recenter and --supersplat are mutually exclusive")

    params = get_internal_params(args.scene_type, args.border_strictness)

    print(f"\nLoading PLY:\n{args.input}")
    ply = PlyData.read(str(args.input))
    if "vertex" not in ply:
        raise RuntimeError("No vertex element found")

    vertices = ply["vertex"].data
    total = len(vertices)
    keep_mask = np.ones(total, dtype=bool)

    print(f"Input splats: {total:,}")

    keep = opacity_filter(vertices, args.min_opacity)
    removed = int(np.sum(keep_mask & ~keep))
    keep_mask &= keep
    print_stage("Opacity filter")
    print(f"min opacity : {args.min_opacity}")
    print(f"removed     : {removed:,}")
    print(f"remaining   : {np.sum(keep_mask):,}")

    keep, scale_thr = global_scale_filter(vertices, args.max_scale_percentile, "max")
    removed = int(np.sum(keep_mask & ~keep))
    keep_mask &= keep
    print_stage("Global scale filter")
    if scale_thr is None:
        print("Skipped: no scale fields found")
    else:
        print("metric      : max")
        print(f"percentile  : {args.max_scale_percentile}")
        print(f"threshold   : {scale_thr:.6f}")
        print(f"removed     : {removed:,}")
        print(f"remaining   : {np.sum(keep_mask):,}")

    current = vertices[keep_mask]
    xyz_current = get_xyz(current)
    plane_normal, plane_info = estimate_scene_plane_normal(xyz_current)

    print_stage("Scene plane estimation")
    print(f"distance mode    : {args.distance_mode}")
    print(f"plane normal     : [{plane_normal[0]:.6f}, {plane_normal[1]:.6f}, {plane_normal[2]:.6f}]")
    print(
        f"eigenvalues      : "
        f"[{plane_info['eigenvalues'][0]:.6f}, {plane_info['eigenvalues'][1]:.6f}, {plane_info['eigenvalues'][2]:.6f}]"
    )

    scene_center, center_info = estimate_scene_center(
        current, "max", params, args.distance_mode, plane_normal
    )

    print_stage("Scene center estimation")
    print(f"scene center     : [{scene_center[0]:.6f}, {scene_center[1]:.6f}, {scene_center[2]:.6f}]")
    print(f"selected points  : {center_info['selected_points']:,}")
    print(f"scale thr        : {center_info['scale_threshold']:.6f}")
    print(f"opacity thr      : {center_info['opacity_threshold']:.6f}")
    print(f"distance thr     : {center_info['distance_threshold']:.6f}")

    current = vertices[keep_mask]
    keep_local, center_thr = center_filter(
        current, args.center_percentile, scene_center, args.distance_mode, plane_normal
    )

    global_idx = np.flatnonzero(keep_mask)
    new_mask = np.zeros_like(keep_mask)
    new_mask[global_idx[keep_local]] = True
    removed = int(np.sum(keep_mask) - np.sum(new_mask))
    keep_mask = new_mask

    print_stage("Center filter")
    print(f"percentile  : {args.center_percentile}")
    print(f"threshold   : {center_thr:.6f}")
    print(f"center      : [{scene_center[0]:.6f}, {scene_center[1]:.6f}, {scene_center[2]:.6f}]")
    print(f"removed     : {removed:,}")
    print(f"remaining   : {np.sum(keep_mask):,}")

    current = vertices[keep_mask]
    keep_local, info = voxel_cluster_filter(
        current, scene_center, args.voxel_grid_size, params, args.distance_mode, plane_normal
    )

    global_idx = np.flatnonzero(keep_mask)
    new_mask = np.zeros_like(keep_mask)
    new_mask[global_idx[keep_local]] = True
    removed = int(np.sum(keep_mask) - np.sum(new_mask))
    keep_mask = new_mask

    vs = info["voxel_size"]

    print_stage("Adaptive voxel cluster filter")
    print(f"scene type       : {args.scene_type}")
    print(f"border strict.   : {args.border_strictness:.2f}")
    print(f"distance mode    : {args.distance_mode}")
    print(f"grid size        : {args.voxel_grid_size}")
    print(f"occupied voxels  : {info['num_occupied_voxels']:,}")
    print(f"active voxels    : {info['num_active_voxels']:,}")
    print(f"selected voxels  : {info['selected_voxels']:,}")
    print(f"selected points  : {info['selected_points']:,}")
    print(f"voxel size       : [{vs[0]:.6f}, {vs[1]:.6f}, {vs[2]:.6f}]")
    print(f"planar limit     : {info['radial_limit']:.6f}")
    print(f"normal limit     : {info['normal_limit']:.6f}")
    if np.isfinite(info["voxel_scale_limit"]):
        print(f"voxel scale lim  : {info['voxel_scale_limit']:.6f}")
    print(f"stop reason      : {info['stop_reason']}")
    print(f"removed          : {removed:,}")
    print(f"remaining        : {np.sum(keep_mask):,}")

    kept = int(np.sum(keep_mask))
    removed = total - kept

    print("\n========== SUMMARY ==========")
    print(f"Total splats     : {total:,}")
    print(f"Final kept       : {kept:,}")
    print(f"Final removed    : {removed:,}")
    print(f"Retention        : {100.0 * kept / max(total, 1):.2f}%")

    if args.dry_run:
        print("\nDry run enabled.")
        return

    filtered_vertices = vertices[keep_mask].copy()
    xyz_new = get_xyz(filtered_vertices)

    if len(xyz_new) == 0:
        raise RuntimeError("All splats were removed")

    if args.recenter:
        xyz_new -= xyz_new.mean(axis=0)
    elif args.supersplat:
        center = xyz_new.mean(axis=0)
        xyz_new -= center
        scale = np.max(np.linalg.norm(xyz_new, axis=1))
        if scale > 0:
            xyz_new /= scale

    filtered_vertices["x"] = xyz_new[:, 0]
    filtered_vertices["y"] = xyz_new[:, 1]
    filtered_vertices["z"] = xyz_new[:, 2]

    PlyData([PlyElement.describe(filtered_vertices, "vertex")], text=False).write(str(args.output))
    print(f"\nSaved:\n{args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nERROR:")
        traceback.print_exc()
        sys.exit(1)

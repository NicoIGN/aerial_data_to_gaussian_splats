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
        return np.exp(np.column_stack([
            vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]
        ]))
    if all(n in names for n in ("sx", "sy", "sz")):
        return np.column_stack([
            np.abs(vertices["sx"]), np.abs(vertices["sy"]), np.abs(vertices["sz"])
        ])
    return None


def compute_scale_metric(scales, metric):
    if metric == "max":  return np.max(scales, axis=1)
    if metric == "mean": return np.mean(scales, axis=1)
    if metric == "norm": return np.linalg.norm(scales, axis=1)
    raise ValueError(metric)


def get_xyz(vertices):
    return np.column_stack([vertices["x"], vertices["y"], vertices["z"]])


def robust_center(xyz):
    return np.median(xyz, axis=0)


def robust_sigma(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad + 1e-12


def print_stage(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_skip(reason):
    print(f"  [SKIPPED] {reason}")


# ═══════════════════════════════════════════════════════════════
# Estimation plan de scène
# ═══════════════════════════════════════════════════════════════

def estimate_scene_plane_normal(xyz):
    if len(xyz) == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), {}
    center  = np.median(xyz, axis=0)
    cov     = np.cov((xyz - center).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order   = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    normal  = eigvecs[:, 0]
    normal  = normal / (np.linalg.norm(normal) + 1e-12)
    return normal, {"plane_center": center, "eigenvalues": eigvals, "normal": normal}


def split_plane_normal_components(points, center, plane_normal=None):
    diff = points - center
    if plane_normal is None:
        return np.linalg.norm(diff, axis=-1), np.zeros(len(points))
    n      = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
    signed = np.sum(diff * n, axis=-1)
    planar = diff - signed[..., None] * n
    return np.linalg.norm(planar, axis=-1), np.abs(signed)


# ═══════════════════════════════════════════════════════════════
# Paramètres internes
# ═══════════════════════════════════════════════════════════════

def get_internal_params(scene_type, border_strictness):
    s     = float(np.clip(border_strictness, 0.0, 1.0))
    s_eff = 0.7 * s + 0.3 * (s ** 2)

    if scene_type == "compact":
        base = dict(
            center_scale_percentile=75.0, center_opacity_percentile=75.0,
            center_keep_percentile=82.0,  voxel_min_points=3,
            voxel_center_sigma=2.4,       voxel_normal_sigma=6.0,
            voxel_max_fraction=0.40,      attach_layers=1,
            attach_min_points=2,          voxel_scale_percentile=95.0,
        )
        radial_density_bias        = 0.3 + 4.0 * s_eff
        scale_bias                 = 0.3 + 5.0 * s_eff
        attach_radial_density_bias = 0.2 + 2.5 * s_eff
        voxel_center_sigma         = max(1.8,  base["voxel_center_sigma"]     - 0.9 * s_eff)
        voxel_scale_percentile     = max(86.0, base["voxel_scale_percentile"] - 7.0 * s_eff)
        voxel_max_fraction         = max(0.22, base["voxel_max_fraction"]     - 0.30 * s_eff)
        if   s_eff >= 0.9:  attach_layers = 0
        elif s_eff >= 0.65: attach_layers = max(0, base["attach_layers"] - 1)
        else:               attach_layers = base["attach_layers"]
        attach_min_points = base["attach_min_points"] + int(s_eff >= 0.75)
        voxel_min_points  = base["voxel_min_points"]  + int(s_eff >= 0.9)

    elif scene_type == "wide":
        base = dict(
            center_scale_percentile=99.0, center_opacity_percentile=45.0,
            center_keep_percentile=99.0,  voxel_min_points=1,
            voxel_center_sigma=4.5,       voxel_normal_sigma=12.0,
            voxel_max_fraction=1.00,      attach_layers=5,
            attach_min_points=1,          voxel_scale_percentile=99.8,
        )
        radial_density_bias        = 0.0 + 0.5 * s_eff
        scale_bias                 = 0.0 + 0.3 * s_eff
        attach_radial_density_bias = 0.0 + 0.3 * s_eff
        voxel_center_sigma         = max(4.0,  base["voxel_center_sigma"]     - 0.2 * s_eff)
        voxel_scale_percentile     = max(99.0, base["voxel_scale_percentile"] - 0.5 * s_eff)
        voxel_max_fraction         = 1.00
        attach_layers              = base["attach_layers"]
        attach_min_points          = 1
        voxel_min_points           = 1

    else:  # balanced
        base = dict(
            center_scale_percentile=85.0, center_opacity_percentile=70.0,
            center_keep_percentile=88.0,  voxel_min_points=2,
            voxel_center_sigma=3.0,       voxel_normal_sigma=7.0,
            voxel_max_fraction=0.70,      attach_layers=2,
            attach_min_points=1,          voxel_scale_percentile=96.0,
        )
        radial_density_bias        = 0.3 + 4.0 * s_eff
        scale_bias                 = 0.3 + 5.0 * s_eff
        attach_radial_density_bias = 0.2 + 2.5 * s_eff
        voxel_center_sigma         = max(1.8,  base["voxel_center_sigma"]     - 0.9 * s_eff)
        voxel_scale_percentile     = max(86.0, base["voxel_scale_percentile"] - 7.0 * s_eff)
        voxel_max_fraction         = max(0.22, base["voxel_max_fraction"]     - 0.30 * s_eff)
        if   s_eff >= 0.9:  attach_layers = 0
        elif s_eff >= 0.65: attach_layers = max(0, base["attach_layers"] - 1)
        else:               attach_layers = base["attach_layers"]
        attach_min_points = base["attach_min_points"] + int(s_eff >= 0.75)
        voxel_min_points  = base["voxel_min_points"]  + int(s_eff >= 0.9)

    return dict(
        center_scale_percentile    = base["center_scale_percentile"],
        center_opacity_percentile  = base["center_opacity_percentile"],
        center_keep_percentile     = base["center_keep_percentile"],
        voxel_min_points           = voxel_min_points,
        voxel_center_sigma         = voxel_center_sigma,
        voxel_normal_sigma         = base["voxel_normal_sigma"],
        voxel_max_fraction         = voxel_max_fraction,
        attach_layers              = attach_layers,
        attach_min_points          = attach_min_points,
        voxel_scale_percentile     = voxel_scale_percentile,
        voxel_radial_density_bias  = radial_density_bias,
        voxel_scale_bias           = scale_bias,
        attach_radial_density_bias = attach_radial_density_bias,
    )


# ═══════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════

def validate_args(args):
    if args.min_opacity is not None and not (0.0 <= args.min_opacity <= 1.0):
        raise ValueError("--min-opacity doit être dans [0, 1]")
    if args.max_scale_percentile is not None and not (0.0 < args.max_scale_percentile <= 100.0):
        raise ValueError("--max-scale-percentile doit être dans (0, 100]")
    if args.max_aspect_ratio is not None and args.max_aspect_ratio < 1.0:
        raise ValueError("--max-aspect-ratio doit être >= 1.0")
    if args.center_percentile is not None and not (0.0 < args.center_percentile <= 100.0):
        raise ValueError("--center-percentile doit être dans (0, 100]")
    if args.voxel_grid_size is not None and args.voxel_grid_size < 8:
        raise ValueError("--voxel-grid-size doit être >= 8")
    if not (0.0 <= args.border_strictness <= 1.0):
        raise ValueError("--border-strictness doit être dans [0, 1]")


# ═══════════════════════════════════════════════════════════════
# Filtres
# ═══════════════════════════════════════════════════════════════

def opacity_filter(vertices, min_opacity):
    if "opacity" not in vertices.dtype.names:
        print("  [WARN] Pas de champ 'opacity' → filtre skippé")
        return np.ones(len(vertices), dtype=bool)
    return decode_opacity(vertices["opacity"]) >= min_opacity


def global_scale_filter(vertices, percentile, metric):
    scales = decode_scales(vertices)
    if scales is None:
        print("  [WARN] Pas de champs d'échelle → filtre skippé")
        return np.ones(len(vertices), dtype=bool), None
    scale_metric = compute_scale_metric(scales, metric)
    threshold    = np.percentile(scale_metric, percentile)
    return scale_metric <= threshold, threshold


def aspect_ratio_filter(vertices, max_aspect_ratio):
    """
    Filtre sur le ratio d'élongation de chaque splat.

    ratio = scale_max / scale_min

    Un splat sphérique a ratio ≈ 1.
    Un splat très allongé (aiguille) ou très plat (crêpe) a un ratio élevé.

    --max-aspect-ratio 10  → supprime les splats avec scale_max > 10 × scale_min
    """
    scales = decode_scales(vertices)
    if scales is None:
        print("  [WARN] Pas de champs d'échelle → filtre aspect ratio skippé")
        return np.ones(len(vertices), dtype=bool), None, None

    scale_max = np.max(scales, axis=1)
    scale_min = np.min(scales, axis=1)
    ratio     = scale_max / (scale_min + 1e-12)

    keep = ratio <= max_aspect_ratio

    # Stats pour le log
    stats = {
        "ratio_min":    float(ratio.min()),
        "ratio_median": float(np.median(ratio)),
        "ratio_p95":    float(np.percentile(ratio, 95)),
        "ratio_p99":    float(np.percentile(ratio, 99)),
        "ratio_max":    float(ratio.max()),
    }
    return keep, ratio, stats


def estimate_scene_center(vertices, scale_metric_name, params, distance_mode, plane_normal):
    xyz = get_xyz(vertices)
    n   = len(xyz)
    if n == 0:
        return np.zeros(3, dtype=np.float64), {}

    base_center = robust_center(xyz)
    if distance_mode == "planar-aware" and plane_normal is not None:
        base_dist, _ = split_plane_normal_components(xyz, base_center, plane_normal)
    else:
        base_dist = np.linalg.norm(xyz - base_center, axis=1)

    opacity = decode_opacity(vertices["opacity"]) if "opacity" in vertices.dtype.names \
              else np.ones(n, dtype=np.float64)
    scales = decode_scales(vertices)
    scale_metric = compute_scale_metric(scales, scale_metric_name) if scales is not None \
                   else np.ones(n, dtype=np.float64)

    scale_thr   = np.percentile(scale_metric, params["center_scale_percentile"])
    opacity_thr = np.percentile(opacity,      params["center_opacity_percentile"])
    dist_thr    = np.percentile(base_dist,    params["center_keep_percentile"])

    keep = (scale_metric <= scale_thr) & (opacity >= opacity_thr) & (base_dist <= dist_thr)
    if np.sum(keep) < max(128, int(0.001 * n)):
        keep = base_dist <= np.percentile(base_dist, 85.0)

    center = base_center if np.sum(keep) == 0 else np.median(xyz[keep], axis=0)
    return center, dict(
        selected_points    = int(np.sum(keep)),
        scale_threshold    = float(scale_thr),
        opacity_threshold  = float(opacity_thr),
        distance_threshold = float(dist_thr),
        center             = center,
    )


def center_filter(vertices, percentile, scene_center, distance_mode, plane_normal):
    xyz = get_xyz(vertices)
    if distance_mode == "planar-aware" and plane_normal is not None:
        dist, _ = split_plane_normal_components(xyz, scene_center, plane_normal)
    else:
        dist = np.linalg.norm(xyz - scene_center, axis=1)
    threshold = np.percentile(dist, percentile)
    return dist <= threshold, threshold


def make_neighbor_offsets():
    return [(dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == dy == dz == 0)]


def voxel_cluster_filter(vertices, scene_center, voxel_grid_size, params, distance_mode, plane_normal):
    xyz = get_xyz(vertices)
    n   = len(xyz)
    if n == 0:
        return np.zeros(0, dtype=bool), {}

    mins   = xyz.min(axis=0)
    maxs   = xyz.max(axis=0)
    extent = np.maximum(maxs - mins, 1e-9)
    aspect = extent.max() / extent.min()
    print(f"\n  [DIAG] Étendue XYZ   : X={extent[0]:.6f}  Y={extent[1]:.6f}  Z={extent[2]:.6f}")
    print(f"  [DIAG] Ratio aspect  : {aspect:.1f}x  {'⚠ scène plate' if aspect > 10 else 'OK'}")

    if distance_mode == "planar-aware" and plane_normal is not None:
        dist_ref, point_normal_dist = split_plane_normal_components(xyz, scene_center, plane_normal)
        normal_limit = np.median(point_normal_dist) + params["voxel_normal_sigma"] * robust_sigma(point_normal_dist)
    else:
        dist_ref          = np.linalg.norm(xyz - scene_center, axis=1)
        point_normal_dist = np.zeros_like(dist_ref)
        normal_limit      = np.inf

    dist_med     = np.median(dist_ref)
    dist_sig     = robust_sigma(dist_ref)
    radial_limit = dist_med + params["voxel_center_sigma"] * dist_sig
    print(f"\n  [DIAG] Radial : median={dist_med:.6f}  sigma={dist_sig:.6f}  "
          f"limit={radial_limit:.6f}  "
          f"({int(np.sum(dist_ref <= radial_limit)):,}/{n:,} pts dans limite)")

    # Grille anisotrope — voxels cubiques en espace monde
    voxel_edge    = extent.max() / float(voxel_grid_size)
    grid_per_axis = np.maximum(np.round(extent / voxel_edge).astype(np.int32), 1)
    voxel_size    = extent / grid_per_axis.astype(np.float64)

    print(f"\n  [DIAG] Grille anisotrope :")
    print(f"         voxel_edge    : {voxel_edge:.6f} m")
    print(f"         grid_per_axis : X={grid_per_axis[0]}  Y={grid_per_axis[1]}  Z={grid_per_axis[2]}")
    print(f"         voxel_size    : X={voxel_size[0]:.6f}  Y={voxel_size[1]:.6f}  Z={voxel_size[2]:.6f}")

    voxel_idx = np.clip(
        np.floor((xyz - mins) / voxel_size).astype(np.int32),
        0, grid_per_axis - 1
    )
    voxel_keys, inverse, counts = np.unique(voxel_idx, axis=0, return_inverse=True, return_counts=True)
    print(f"\n  [DIAG] Voxels occupés : {len(voxel_keys):,}  "
          f"(médiane pts/voxel={np.median(counts):.1f}  max={counts.max()})")

    voxel_centers = mins + (voxel_keys.astype(np.float64) + 0.5) * voxel_size

    if distance_mode == "planar-aware" and plane_normal is not None:
        voxel_center_dist, voxel_normal_dist = split_plane_normal_components(
            voxel_centers, scene_center, plane_normal)
    else:
        voxel_center_dist = np.linalg.norm(voxel_centers - scene_center, axis=1)
        voxel_normal_dist = np.zeros_like(voxel_center_dist)

    voxel_radial_norm = voxel_center_dist / (radial_limit + 1e-12)

    scales = decode_scales(vertices)
    if scales is not None:
        point_scale_metric = compute_scale_metric(scales, "max")
        voxel_scale        = np.zeros(len(voxel_keys), dtype=np.float64)
        order  = np.argsort(inverse)
        inv_s  = inverse[order]
        sc_s   = point_scale_metric[order]
        starts = np.concatenate(([0], np.flatnonzero(np.diff(inv_s)) + 1, [len(order)]))
        for i in range(len(starts) - 1):
            a, b = starts[i], starts[i + 1]
            voxel_scale[inv_s[a]] = np.median(sc_s[a:b])
        voxel_scale_limit = np.percentile(voxel_scale, params["voxel_scale_percentile"])
        print(f"\n  [DIAG] Échelle voxel : min={voxel_scale.min():.6f}  "
              f"median={np.median(voxel_scale):.6f}  max={voxel_scale.max():.6f}  "
              f"limit={voxel_scale_limit:.6f}")
    else:
        voxel_scale       = np.ones(len(voxel_keys), dtype=np.float64)
        voxel_scale_limit = np.inf

    dense_required = params["voxel_min_points"] * (
        1.0 + params["voxel_radial_density_bias"] * (voxel_radial_norm ** 2))

    if np.isfinite(voxel_scale_limit):
        penalty       = np.minimum(
            params["voxel_scale_bias"] * np.maximum(voxel_radial_norm - 0.5, 0.0) ** 2, 1.0)
        scale_allowed = voxel_scale_limit / (1.0 + penalty)
        scale_ok      = voxel_scale <= scale_allowed
    else:
        scale_ok = np.ones(len(voxel_keys), dtype=bool)

    radial_ok = voxel_center_dist <= radial_limit
    normal_ok = voxel_normal_dist <= normal_limit
    dense_ok  = counts >= dense_required

    print(f"\n  [DIAG] Filtres voxel :")
    print(f"         dense_ok  : {int(dense_ok.sum()):,} / {len(voxel_keys):,}  "
          f"(min_points={params['voxel_min_points']})")
    print(f"         radial_ok : {int(radial_ok.sum()):,} / {len(voxel_keys):,}")
    print(f"         normal_ok : {int(normal_ok.sum()):,} / {len(voxel_keys):,}")
    print(f"         scale_ok  : {int(scale_ok.sum()):,} / {len(voxel_keys):,}")
    print(f"         Tous OK   : {int((dense_ok & radial_ok & normal_ok & scale_ok).sum()):,}")
    print(f"  [DIAG] Goulots :")
    print(f"         dense seul  : {int((~dense_ok &  radial_ok &  normal_ok &  scale_ok).sum()):,}")
    print(f"         scale seul  : {int(( dense_ok &  radial_ok &  normal_ok & ~scale_ok).sum()):,}")
    print(f"         radial seul : {int(( dense_ok & ~radial_ok &  normal_ok &  scale_ok).sum()):,}")

    active_mask        = dense_ok & radial_ok & normal_ok & scale_ok
    active_voxel_keys  = voxel_keys[active_mask]
    active_center_dist = voxel_center_dist[active_mask]

    if len(active_voxel_keys) == 0:
        return np.zeros(n, dtype=bool), dict(
            num_occupied_voxels=len(voxel_keys), num_active_voxels=0,
            selected_voxels=0, selected_points=0,
            radial_limit=radial_limit, normal_limit=normal_limit,
            voxel_scale_limit=voxel_scale_limit, voxel_size=voxel_size,
            stop_reason="no_active_voxels")

    active_dict = {tuple(v.tolist()): i for i, v in enumerate(active_voxel_keys)}
    seed_idx    = int(np.argmin(active_center_dist))
    seed_voxel  = tuple(active_voxel_keys[seed_idx].tolist())
    offsets     = make_neighbor_offsets()
    selected    = {seed_voxel}
    q           = deque([seed_voxel])
    max_voxels  = max(1, int(np.floor(params["voxel_max_fraction"] * len(active_voxel_keys))))
    stop_reason = "frontier_exhausted"

    print(f"\n  [DIAG] BFS : actifs={len(active_voxel_keys):,}  max={max_voxels}  graine={seed_voxel}")

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

    pct = 100 * len(selected) / len(active_voxel_keys)
    print(f"         BFS : {len(selected):,} voxels atteints ({pct:.1f}%)  [{stop_reason}]")
    if pct < 50 and stop_reason == "frontier_exhausted":
        print(f"         ⚠  Grille disconnectée")
    else:
        print(f"         ✓  Grille connectée")

    if params["attach_layers"] > 0:
        occupied_dict     = {tuple(v.tolist()): i for i, v in enumerate(voxel_keys)}
        frontier          = set(selected)
        selected_expanded = set(selected)
        for layer in range(params["attach_layers"]):
            new_frontier = set()
            for v in frontier:
                x, y, z = v
                for dx, dy, dz in offsets:
                    nb = (x + dx, y + dy, z + dz)
                    if nb in selected_expanded or nb not in occupied_dict:
                        continue
                    occ_idx     = occupied_dict[nb]
                    radial_norm = voxel_radial_norm[occ_idx]
                    req         = params["attach_min_points"] * (
                        1.0 + params["attach_radial_density_bias"] * radial_norm ** 2)
                    if counts[occ_idx]            < req:           continue
                    if voxel_center_dist[occ_idx] > radial_limit:  continue
                    if voxel_normal_dist[occ_idx] > normal_limit:  continue
                    if not scale_ok[occ_idx]:                      continue
                    selected_expanded.add(nb)
                    new_frontier.add(nb)
            print(f"         Couche attach {layer+1} : +{len(new_frontier)} voxels")
            if not new_frontier:
                break
            frontier = new_frontier
        selected = selected_expanded

    selected_struct = set(selected)
    keep_points = np.array([tuple(v.tolist()) in selected_struct for v in voxel_idx], dtype=bool)
    print(f"\n  [DIAG] Points conservés : {int(keep_points.sum()):,} / {n:,} "
          f"({100*keep_points.sum()/n:.1f}%)")

    return keep_points, dict(
        num_occupied_voxels = len(voxel_keys),
        num_active_voxels   = len(active_voxel_keys),
        selected_voxels     = len(selected),
        selected_points     = int(np.sum(keep_points)),
        radial_limit        = radial_limit,
        normal_limit        = normal_limit,
        voxel_scale_limit   = voxel_scale_limit,
        voxel_size          = voxel_size,
        stop_reason         = stop_reason,
        seed_voxel          = seed_voxel,
    )


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Nettoie un PLY Gaussian Splat. "
                    "Chaque filtre est optionnel : si son paramètre est absent, il est skippé."
    )
    p.add_argument("--input",  required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)

    p.add_argument("--min-opacity",          type=float, default=None,
                   help="Opacité minimum [0,1]. Absent = skippé.")
    p.add_argument("--max-scale-percentile", type=float, default=None,
                   help="Percentile max d'échelle globale (0,100]. Absent = skippé.")
    p.add_argument("--max-aspect-ratio",     type=float, default=None,
                   help="Ratio max scale_max/scale_min par splat (≥1). "
                        "Ex: 10 → supprime les splats 10× plus longs que larges. "
                        "Absent = skippé.")
    p.add_argument("--center-percentile",    type=float, default=None,
                   help="Percentile distance au centre (0,100]. Absent = skippé.")
    p.add_argument("--voxel-grid-size",      type=int,   default=None,
                   help="Taille grille voxel (≥8). Absent = filtre voxel skippé.")

    p.add_argument("--scene-type",        choices=["compact", "balanced", "wide"], default="balanced")
    p.add_argument("--border-strictness", type=float, default=0.5)
    p.add_argument("--distance-mode",     choices=["euclidean", "planar-aware"],   default="planar-aware")
    p.add_argument("--dry-run",           action="store_true")
    p.add_argument("--recenter",          action="store_true")
    p.add_argument("--supersplat",        action="store_true")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    validate_args(args)

    if args.recenter and args.supersplat:
        raise ValueError("--recenter et --supersplat sont mutuellement exclusifs")

    params = get_internal_params(args.scene_type, args.border_strictness) \
             if args.voxel_grid_size is not None else None

    print(f"\nChargement PLY :\n{args.input}")
    ply = PlyData.read(str(args.input))
    if "vertex" not in ply:
        raise RuntimeError("Pas d'élément 'vertex' trouvé")

    vertices  = ply["vertex"].data
    total     = len(vertices)
    keep_mask = np.ones(total, dtype=bool)
    print(f"Splats en entrée : {total:,}")

    # ── Résumé des filtres actifs ─────────────────────────────── #
    print_stage("Filtres actifs")
    def fmt(v): return str(v) if v is not None else "— skippé"
    print(f"  --min-opacity          : {fmt(args.min_opacity)}")
    print(f"  --max-scale-percentile : {fmt(args.max_scale_percentile)}")
    print(f"  --max-aspect-ratio     : {fmt(args.max_aspect_ratio)}")
    print(f"  --center-percentile    : {fmt(args.center_percentile)}")
    print(f"  --voxel-grid-size      : {fmt(args.voxel_grid_size)}")
    if args.voxel_grid_size is not None:
        print(f"  --scene-type           : {args.scene_type}")
        print(f"  --border-strictness    : {args.border_strictness}")
        print(f"  --distance-mode        : {args.distance_mode}")

    # ══════════════════════════════════════════════════════════════
    # FILTRE 1 — Opacité
    # ══════════════════════════════════════════════════════════════
    print_stage("Filtre 1 — Opacité")
    if args.min_opacity is None:
        print_skip("--min-opacity non fourni")
    else:
        keep    = opacity_filter(vertices, args.min_opacity)
        removed = int(np.sum(keep_mask & ~keep))
        keep_mask &= keep
        print(f"  seuil     : {args.min_opacity}")
        print(f"  supprimés : {removed:,}")
        print(f"  restants  : {np.sum(keep_mask):,}")

    # ══════════════════════════════════════════════════════════════
    # FILTRE 2 — Échelle globale
    # ══════════════════════════════════════════════════════════════
    print_stage("Filtre 2 — Échelle globale (percentile)")
    if args.max_scale_percentile is None:
        print_skip("--max-scale-percentile non fourni")
    else:
        keep, scale_thr = global_scale_filter(vertices, args.max_scale_percentile, "max")
        removed = int(np.sum(keep_mask & ~keep))
        keep_mask &= keep
        if scale_thr is None:
            print("  [WARN] Pas de champs d'échelle dans le PLY")
        else:
            print(f"  percentile : {args.max_scale_percentile}")
            print(f"  seuil      : {scale_thr:.6f}")
            print(f"  supprimés  : {removed:,}")
            print(f"  restants   : {np.sum(keep_mask):,}")

    # ══════════════════════════════════════════════════════════════
    # FILTRE 3 — Ratio d'aspect (élongation)
    # ══════════════════════════════════════════════════════════════
    print_stage("Filtre 3 — Ratio d'aspect (scale_max / scale_min)")
    if args.max_aspect_ratio is None:
        print_skip("--max-aspect-ratio non fourni")
    else:
        keep, ratio, stats = aspect_ratio_filter(vertices, args.max_aspect_ratio)
        if ratio is None:
            print("  [WARN] Pas de champs d'échelle dans le PLY")
        else:
            removed = int(np.sum(keep_mask & ~keep))
            keep_mask &= keep
            print(f"  seuil max   : {args.max_aspect_ratio:.2f}")
            print(f"  ratio min   : {stats['ratio_min']:.2f}")
            print(f"  ratio median: {stats['ratio_median']:.2f}")
            print(f"  ratio p95   : {stats['ratio_p95']:.2f}")
            print(f"  ratio p99   : {stats['ratio_p99']:.2f}")
            print(f"  ratio max   : {stats['ratio_max']:.2f}")
            print(f"  supprimés   : {removed:,}")
            print(f"  restants    : {np.sum(keep_mask):,}")

    # ══════════════════════════════════════════════════════════════
    # Estimation plan + centre (si nécessaire)
    # ══════════════════════════════════════════════════════════════
    need_center  = (args.center_percentile is not None) or (args.voxel_grid_size is not None)
    plane_normal = None
    scene_center = None

    if need_center:
        current      = vertices[keep_mask]
        plane_normal, plane_info = estimate_scene_plane_normal(get_xyz(current))

        print_stage("Estimation plan de scène")
        print(f"  normal      : [{plane_normal[0]:.6f}, {plane_normal[1]:.6f}, {plane_normal[2]:.6f}]")
        ev    = plane_info['eigenvalues']
        ratio = ev[2] / (ev[0] + 1e-12)
        print(f"  eigenvalues : [{ev[0]:.6f}, {ev[1]:.6f}, {ev[2]:.6f}]")
        print(f"  ratio λ     : {ratio:.1f}x  "
              f"{'⚠ scène très plate' if ratio > 50 else 'OK'}")

        _params = params if params else get_internal_params("balanced", 0.5)
        scene_center, center_info = estimate_scene_center(
            current, "max", _params, args.distance_mode, plane_normal)

        print_stage("Estimation centre de scène")
        print(f"  centre         : [{scene_center[0]:.6f}, {scene_center[1]:.6f}, {scene_center[2]:.6f}]")
        print(f"  points sélect. : {center_info['selected_points']:,}")

    # ══════════════════════════════════════════════════════════════
    # FILTRE 4 — Centre
    # ══════════════════════════════════════════════════════════════
    print_stage("Filtre 4 — Distance au centre")
    if args.center_percentile is None:
        print_skip("--center-percentile non fourni")
    else:
        current    = vertices[keep_mask]
        keep_local, center_thr = center_filter(
            current, args.center_percentile, scene_center, args.distance_mode, plane_normal)
        global_idx = np.flatnonzero(keep_mask)
        new_mask   = np.zeros_like(keep_mask)
        new_mask[global_idx[keep_local]] = True
        removed    = int(np.sum(keep_mask) - np.sum(new_mask))
        keep_mask  = new_mask
        print(f"  percentile : {args.center_percentile}")
        print(f"  seuil      : {center_thr:.6f}")
        print(f"  supprimés  : {removed:,}")
        print(f"  restants   : {np.sum(keep_mask):,}")

    # ══════════════════════════════════════════════════════════════
    # FILTRE 5 — Voxel cluster
    # ══════════════════════════════════════════════════════════════
    print_stage("Filtre 5 — Voxel cluster adaptatif")
    if args.voxel_grid_size is None:
        print_skip("--voxel-grid-size non fourni")
    else:
        current    = vertices[keep_mask]
        keep_local, info = voxel_cluster_filter(
            current, scene_center, args.voxel_grid_size,
            params, args.distance_mode, plane_normal)
        global_idx = np.flatnonzero(keep_mask)
        new_mask   = np.zeros_like(keep_mask)
        new_mask[global_idx[keep_local]] = True
        removed    = int(np.sum(keep_mask) - np.sum(new_mask))
        keep_mask  = new_mask
        vs = info["voxel_size"]
        print(f"\n  scene type      : {args.scene_type}")
        print(f"  border strict.  : {args.border_strictness:.2f}")
        print(f"  distance mode   : {args.distance_mode}")
        print(f"  grid size       : {args.voxel_grid_size}")
        print(f"  voxels occupés  : {info['num_occupied_voxels']:,}")
        print(f"  voxels actifs   : {info['num_active_voxels']:,}")
        print(f"  voxels sélect.  : {info['selected_voxels']:,}")
        print(f"  points sélect.  : {info['selected_points']:,}")
        print(f"  voxel size      : [{vs[0]:.6f}, {vs[1]:.6f}, {vs[2]:.6f}]")
        print(f"  planar limit    : {info['radial_limit']:.6f}")
        print(f"  normal limit    : {info['normal_limit']:.6f}")
        if np.isfinite(info["voxel_scale_limit"]):
            print(f"  scale limit     : {info['voxel_scale_limit']:.6f}")
        print(f"  stop reason     : {info['stop_reason']}")
        print(f"  supprimés       : {removed:,}")
        print(f"  restants        : {np.sum(keep_mask):,}")

    # ══════════════════════════════════════════════════════════════
    # Résumé
    # ══════════════════════════════════════════════════════════════
    kept    = int(np.sum(keep_mask))
    removed = total - kept

    print("\n" + "=" * 60)
    print("  RÉSUMÉ FINAL")
    print("=" * 60)
    print(f"  Total splats   : {total:,}")
    print(f"  Conservés      : {kept:,}")
    print(f"  Supprimés      : {removed:,}")
    print(f"  Rétention      : {100.0 * kept / max(total, 1):.2f}%")

    if args.dry_run:
        print("\n  [DRY RUN] Pas d'écriture.")
        return

    filtered_vertices = vertices[keep_mask].copy()
    xyz_new           = get_xyz(filtered_vertices)
    if len(xyz_new) == 0:
        raise RuntimeError("Tous les splats ont été supprimés")

    if args.recenter:
        xyz_new -= xyz_new.mean(axis=0)
    elif args.supersplat:
        center   = xyz_new.mean(axis=0)
        xyz_new -= center
        scale    = np.max(np.linalg.norm(xyz_new, axis=1))
        if scale > 0:
            xyz_new /= scale

    filtered_vertices["x"] = xyz_new[:, 0]
    filtered_vertices["y"] = xyz_new[:, 1]
    filtered_vertices["z"] = xyz_new[:, 2]

    PlyData([PlyElement.describe(filtered_vertices, "vertex")], text=False).write(str(args.output))
    print(f"\n  Sauvegardé :\n  {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nERREUR :")
        traceback.print_exc()
        sys.exit(1)

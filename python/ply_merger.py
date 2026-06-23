#!/usr/bin/env python3
"""
Fusion de N PLY Gaussian Splat.

Pipeline :
    X_world = R_inv @ (T_colmap @ X_nerf / scale_nerf - t_nerf)
    X_out   = (X_world - center) / extent_max   ← re-normalisation pour SuperSplat

Fichiers générés automatiquement à partir du nom du PLY de sortie :
    merged.ply          → PLY fusionné re-normalisé
    merged_report.json  → bbox, stats, pipeline
    merged_transforms.json → infos pour reconstruire les coords world

Usage :
    python ply_merger.py \\
        --inputs dalle1.ply norm1.json nerf1.json \\
                 dalle2.ply norm2.json nerf2.json \\
        --output merged.ply
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import numpy.lib.recfunctions as rfn
from plyfile import PlyData, PlyElement


# ═══════════════════════════════════════════════════════════════
# Transformation géométrique
# ═══════════════════════════════════════════════════════════════

def build_transform(normalize: dict, dataparser: dict):
    T_colmap = np.array(normalize["translation_matrix_4x4"], dtype=np.float64)  # 4×4
    s        = float(dataparser["scale"])
    R        = np.array(dataparser["transform"], dtype=np.float64)[:3, :3]       # 3×3
    t        = np.array(dataparser["transform"], dtype=np.float64)[:3, 3]        # (3,)
    R_inv    = R.T   # rotation pure → inverse = transposée
    return T_colmap, R_inv, t, s


def nerf_to_world(xyz: np.ndarray, T_colmap, R_inv, t, s) -> np.ndarray:
    """(N,3) nerfstudio → (N,3) world coords"""
    n    = len(xyz)
    ones = np.ones((n, 1), dtype=np.float64)
    xyzh = np.hstack([xyz.astype(np.float64), ones])          # (N,4)
    xyz_swapped = (T_colmap @ xyzh.T).T[:, :3]                # swap axes
    xyz_scaled  = xyz_swapped / s                              # dé-scaler
    xyz_world   = (R_inv @ (xyz_scaled - t).T).T              # rotation inverse
    return xyz_world


def rotate_quats(quats: np.ndarray, R3: np.ndarray) -> np.ndarray:
    """(N,4)[w,x,y,z] + R3(3×3) → (N,4)[w,x,y,z] normalisés"""
    w, x, y, z = quats[:,0], quats[:,1], quats[:,2], quats[:,3]
    n = len(quats)
    Rs = np.zeros((n, 3, 3), dtype=np.float64)
    Rs[:,0,0] = 1-2*(y*y+z*z); Rs[:,0,1] = 2*(x*y-z*w); Rs[:,0,2] = 2*(x*z+y*w)
    Rs[:,1,0] = 2*(x*y+z*w);   Rs[:,1,1] = 1-2*(x*x+z*z); Rs[:,1,2] = 2*(y*z-x*w)
    Rs[:,2,0] = 2*(x*z-y*w);   Rs[:,2,1] = 2*(y*z+x*w);   Rs[:,2,2] = 1-2*(x*x+y*y)
    Rs_rot = np.einsum('ij,njk->nik', R3, Rs)
    out = np.zeros((n, 4), dtype=np.float64)
    for i, R in enumerate(Rs_rot):
        tr = R[0,0] + R[1,1] + R[2,2]
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1.0)
            out[i] = [0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s]
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[0,0] - R[1,1] - R[2,2]))
            out[i] = [(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s]
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[1,1] - R[0,0] - R[2,2]))
            out[i] = [(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s]
        else:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[2,2] - R[0,0] - R[1,1]))
            out[i] = [(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.maximum(norms, 1e-12)).astype(np.float32)


def rotate_normals(normals: np.ndarray, R3: np.ndarray) -> np.ndarray:
    """(N,3) → (N,3) : rotation pure sans translation"""
    return (R3 @ normals.astype(np.float64).T).T.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# I/O PLY
# ═══════════════════════════════════════════════════════════════

def read_tile(path: Path) -> np.ndarray:
    ply = PlyData.read(str(path))
    if "vertex" not in [e.name for e in ply.elements]:
        raise RuntimeError(f"Pas d'élément 'vertex' dans {path}")
    return ply["vertex"].data.copy()


# ═══════════════════════════════════════════════════════════════
# Traitement d'une dalle
# ═══════════════════════════════════════════════════════════════

def process_tile(ply_path: Path, norm_path: Path, nerf_path: Path):
    print(f"\n{'─'*55}")
    print(f"  Dalle : {ply_path.name}")
    print(f"{'─'*55}")

    vertices   = read_tile(ply_path)
    normalize  = json.loads(norm_path.read_text())
    dataparser = json.loads(nerf_path.read_text())
    n          = len(vertices)

    print(f"  Points  : {n:,}")
    print(f"  Champs  : {len(vertices.dtype.names)} champs")

    T_colmap, R_inv, t, s = build_transform(normalize, dataparser)
    R_combined = R_inv @ T_colmap[:3, :3]   # rotation complète pour quaternions/normales

    # ── Positions ──────────────────────────────────────────
    xyz       = np.column_stack([vertices["x"], vertices["y"], vertices["z"]])
    xyz_world = nerf_to_world(xyz, T_colmap, R_inv, t, s)
    print(f"  XYZ world min : {xyz_world.min(axis=0)}")
    print(f"  XYZ world max : {xyz_world.max(axis=0)}")

    # ── Quaternions ────────────────────────────────────────
    rot_fields = [f"rot_{i}" for i in range(4)]
    if all(f in vertices.dtype.names for f in rot_fields):
        quats = np.column_stack([vertices[f] for f in rot_fields]).astype(np.float64)
        quats_rot = rotate_quats(quats, R_combined)
        for i in range(4):
            vertices[f"rot_{i}"] = quats_rot[:, i]
        print(f"  ✓ Quaternions transformés")
    else:
        print(f"  [WARN] Pas de rot_0..rot_3")

    # ── Normales ───────────────────────────────────────────
    if all(f in vertices.dtype.names for f in ["nx", "ny", "nz"]):
        normals = np.column_stack([vertices["nx"], vertices["ny"], vertices["nz"]]).astype(np.float64)
        normals_rot = rotate_normals(normals, R_combined)
        vertices["nx"] = normals_rot[:, 0]
        vertices["ny"] = normals_rot[:, 1]
        vertices["nz"] = normals_rot[:, 2]
        print(f"  ✓ Normales transformées")

    meta = {
        "file"        : str(ply_path),
        "points"      : n,
        "bbox_world"  : {
            "min" : xyz_world.min(axis=0).tolist(),
            "max" : xyz_world.max(axis=0).tolist(),
        },
        "median_world": np.median(xyz_world, axis=0).tolist(),
        "origin_world": normalize["origin_world"],
        "scale_nerf"  : s,
    }
    return vertices, xyz_world, meta


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Triplets : dalle.ply normalize.json dataparser.json (répété N fois)")
    p.add_argument("--output", required=True, type=Path,
                   help="PLY de sortie (ex: merged.ply)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Noms de fichiers dérivés du PLY de sortie ──────────
    out_ply        = args.output
    out_report     = out_ply.with_name(out_ply.stem + "_report.json")
    out_transforms = out_ply.with_name(out_ply.stem + "_transforms.json")

    # ── Vérification inputs ────────────────────────────────
    if len(args.inputs) % 3 != 0:
        print(f"ERREUR : --inputs doit contenir des triplets.")
        print(f"         Reçu {len(args.inputs)} fichiers (pas divisible par 3).")
        sys.exit(1)

    triplets = []
    for i in range(0, len(args.inputs), 3):
        trio = (Path(args.inputs[i]), Path(args.inputs[i+1]), Path(args.inputs[i+2]))
        for p in trio:
            if not p.exists():
                print(f"ERREUR : fichier introuvable : {p}")
                sys.exit(1)
        triplets.append(trio)

    print(f"\n{'='*55}")
    print(f"  FUSION DE {len(triplets)} DALLE(S)")
    print(f"{'='*55}")
    print(f"  PLY         → {out_ply}")
    print(f"  Report      → {out_report}")
    print(f"  Transforms  → {out_transforms}")

    # ── Traitement des dalles ──────────────────────────────
    all_vertices  = []
    all_xyz_world = []
    all_meta      = []

    for trio in triplets:
        vertices, xyz_world, meta = process_tile(*trio)
        all_vertices.append(vertices)
        all_xyz_world.append(xyz_world)
        all_meta.append(meta)

    # ── Bbox globale ───────────────────────────────────────
    all_xyz  = np.vstack(all_xyz_world)
    g_min    = all_xyz.min(axis=0)
    g_max    = all_xyz.max(axis=0)
    g_center = (g_min + g_max) / 2.0
    g_extent = g_max - g_min
    g_scale  = float(g_extent.max())   # normalisation isotrope

    print(f"\n{'='*55}")
    print(f"  BBOX GLOBALE (coords world)")
    print(f"{'='*55}")
    print(f"  min     : {g_min}")
    print(f"  max     : {g_max}")
    print(f"  étendue : {g_extent}")
    print(f"  centre  : {g_center}")
    print(f"  scale   : {g_scale:.4f}m  (pour re-normalisation → ~[-0.5, 0.5])")

    # ── Concaténer et re-normaliser ────────────────────────
    merged = rfn.stack_arrays(all_vertices, usemask=False, asrecarray=False)
    total  = len(merged)
    print(f"\n  Total points : {total:,}")

    offset = 0
    for xyz_tile in all_xyz_world:
        n_tile  = len(xyz_tile)
        xyz_out = (xyz_tile - g_center) / g_scale   # → [-0.5, +0.5]
        merged["x"][offset:offset+n_tile] = xyz_out[:, 0].astype(np.float32)
        merged["y"][offset:offset+n_tile] = xyz_out[:, 1].astype(np.float32)
        merged["z"][offset:offset+n_tile] = xyz_out[:, 2].astype(np.float32)
        offset += n_tile

    print(f"  XYZ normalisé min : {merged['x'].min():.4f} {merged['y'].min():.4f} {merged['z'].min():.4f}")
    print(f"  XYZ normalisé max : {merged['x'].max():.4f} {merged['y'].max():.4f} {merged['z'].max():.4f}")

    # ── Écrire PLY ─────────────────────────────────────────
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(merged, "vertex")], text=False).write(str(out_ply))
    print(f"\n  ✓ PLY         : {out_ply}  ({total:,} points)")

    # ── Transforms JSON ────────────────────────────────────
    # Permet de reconstruire X_world depuis X_ply :
    #   X_world = X_ply * g_scale + g_center
    transforms = {
        "pipeline"   : "X_world = R_inv @ (T_colmap @ X_nerf / scale_nerf - t_nerf)",
        "normalization": {
            "formula"  : "X_ply = (X_world - center) / scale",
            "formula_inv": "X_world = X_ply * scale + center",
            "center"   : g_center.tolist(),
            "scale"    : g_scale,
            "extent_world": g_extent.tolist(),
        },
        "bbox_world" : {
            "min"    : g_min.tolist(),
            "max"    : g_max.tolist(),
        },
        "tiles": [
            {
                "file"        : m["file"],
                "origin_world": m["origin_world"],
                "scale_nerf"  : m["scale_nerf"],
            }
            for m in all_meta
        ],
    }
    out_transforms.write_text(json.dumps(transforms, indent=2))
    print(f"  ✓ Transforms  : {out_transforms}")

    # ── Report JSON ────────────────────────────────────────
    report = {
        "total_points": total,
        "nb_tiles"    : len(triplets),
        "bbox_world"  : {
            "min"    : g_min.tolist(),
            "max"    : g_max.tolist(),
            "extent" : g_extent.tolist(),
        },
        "normalization": {
            "center" : g_center.tolist(),
            "scale"  : g_scale,
            "note"   : "X_world = X_ply * scale + center",
        },
        "tiles": all_meta,
    }
    out_report.write_text(json.dumps(report, indent=2))
    print(f"  ✓ Report      : {out_report}")

    print(f"\n{'='*55}")
    print(f"  TERMINÉ")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nERREUR :")
        traceback.print_exc()
        sys.exit(1)

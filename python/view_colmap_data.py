#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
from pathlib import Path

try:
    import open3d as o3d
except ImportError:
    print("Erreur: open3d requis. Installez avec : pip install open3d")
    raise

try:
    from PIL import Image
except ImportError:
    print("Erreur: pillow requis. Installez avec : pip install pillow")
    raise


def load_json(path: Path):
    """Charge le fichier JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_T_wc_from_frame(frame):
    """Construit la matrice monde -> caméra à partir des données de `transforms.json`."""
    if "transform_matrix" in frame:
        T_wc = np.array(frame["transform_matrix"], dtype=np.float64)
        if T_wc.shape != (4, 4):
            raise ValueError("transform_matrix doit être 4x4")
        return T_wc
    raise ValueError("Frame invalide : manque transform_matrix")


def create_camera_frustum(T_wc, scale=5.0, color=(1.0, 0.0, 0.0)):
    """
    Crée un frustum de caméra orienté dans l'espace 3D.
    """
    pts_cam = np.array([
        [0.0, 0.0, 0.0],  # Point d'origine de la caméra
        [-0.5, -0.3, 1.0],
        [0.5, -0.3, 1.0],
        [0.5, 0.3, 1.0],
        [-0.5, 0.3, 1.0],
    ], dtype=np.float64)
    pts_cam *= scale

    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3, 3]
    pts_world = (R_wc @ pts_cam.T).T + t_wc

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Origine caméra vers les coins
        [1, 2], [2, 3], [3, 4], [4, 1]   # Connexion des coins formant le rectangle
    ]
    colors = np.tile(np.array(color, dtype=np.float64), (len(lines), 1))

    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(pts_world)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector(colors)
    return frustum


def load_lidar_ply(colmap_dir: Path):
    """
    Charge un fichier nuage de points COLMAP (prend en compte différents noms comme `sparse_pc.ply`, `points.ply`).
    """
    candidate_paths = [
        colmap_dir / "points.ply",
        colmap_dir / "sparse_pc.ply",  # Exemple dans ton répertoire
    ]

    for path in candidate_paths:
        if path.exists():
            print(f"[INFO] Chargement du nuage de points : {path}")
            return o3d.io.read_point_cloud(str(path))

    raise FileNotFoundError(
        f"Aucun fichier de nuage de points valide trouvé dans {colmap_dir}. "
        f"Recherchés : {[str(p) for p in candidate_paths]}"
    )


def visualize_colmap(frames, lidar, scale_axes=3.0, scale_frustums=5.0):
    """
    Affiche les caméras, points 3D et images associées dans un viewer Open3D.
    """
    geoms = [lidar]  # Ajoute d'abord le nuage de points

    print("[INFO] Création des frustums et axes pour les caméras...")
    for frame in frames:
        T_wc = build_T_wc_from_frame(frame)

        # Ajoute les frustums
        frustum = create_camera_frustum(T_wc, scale=scale_frustums, color=(1.0, 0.0, 0.0))
        geoms.append(frustum)

    print("[INFO] Lancement du viewer...")
    o3d.visualization.draw_geometries(geoms, window_name="COLMAP Viewer")


def main():
    """Point d'entrée du script."""
    ap = argparse.ArgumentParser(description="Affiche les données COLMAP dans un viewer 3D")
    ap.add_argument("--colmap-dir", required=True, help="Répertoire contenant transforms.json et le nuage de points")
    args = ap.parse_args()

    colmap_dir = Path(args.colmap_dir)
    transforms_path = colmap_dir / "transforms.json"

    if not transforms_path.exists():
        raise FileNotFoundError(f"Fichier 'transforms.json' introuvable dans {transforms_path}")

    print("[INFO] Chargement des données JSON...")
    data = load_json(transforms_path)
    frames = data.get("frames", [])
    if not frames:
        raise ValueError("Aucune caméra ou frame valide détectée dans transforms.json")

    lidar = load_lidar_ply(colmap_dir)
    visualize_colmap(frames, lidar)


if __name__ == "__main__":
    main()

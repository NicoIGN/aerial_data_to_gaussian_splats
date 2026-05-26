#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

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
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_T_wc_from_frame(frame):
    if "transform_matrix" in frame:
        T_wc = np.array(frame["transform_matrix"], dtype=np.float64)
        if T_wc.shape != (4, 4):
            raise ValueError("transform_matrix doit être 4x4")
        return T_wc
    raise ValueError("Frame invalide : manque transform_matrix")


def create_camera_frustum(T_wc, scale=12.0, color=(1.0, 0.0, 0.0)):
    pts_cam = np.array([
        [0.0, 0.0, 0.0],
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
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]
    colors = np.tile(np.array(color, dtype=np.float64), (len(lines), 1))

    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(pts_world)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector(colors)
    return frustum


def create_camera_axes(T_wc, axis_size=3.0):
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
    mesh.transform(T_wc)
    return mesh


def load_lidar_ply(colmap_dir: Path):
    candidate_paths = [
        colmap_dir / "points.ply",
        colmap_dir / "sparse_pc.ply",
    ]

    for path in candidate_paths:
        if path.exists():
            print(f"[INFO] Chargement du nuage de points : {path}")
            return o3d.io.read_point_cloud(str(path))

    raise FileNotFoundError(
        f"Aucun fichier de nuage de points valide trouvé dans {colmap_dir}. "
        f"Recherchés : {[str(p) for p in candidate_paths]}"
    )


def preview_cache_path(cache_dir: Path, image_path: Path):
    return cache_dir / f"{image_path.stem}.jpg"


def run_make_preview_script(make_preview_script: Path, src_path: Path, cache_dir: Path,
                            max_size=1024, verbose=False):
    cmd = [
        sys.executable,
        str(make_preview_script),
        "--input", str(src_path),
        "--output-dir", str(cache_dir),
        "--max-size", str(max_size),
        "--jpg",
    ]

    if verbose:
        cmd.append("--verbose")
        print(f"[INFO] Exécution: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=not verbose, text=True)

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        stdout = result.stdout.strip() if result.stdout else ""
        raise RuntimeError(
            f"Échec de make_preview.py pour {src_path}\n"
            f"stdout: {stdout}\n"
            f"stderr: {stderr}"
        )


def get_preview_image(frame, colmap_dir: Path, cache_dir: Path,
                      make_preview_script: Path, max_size=1024, verbose=False):
    file_path = frame.get("file_path")
    if not file_path:
        return None

    src_path = colmap_dir / file_path
    if not src_path.exists():
        print(f"[WARN] Image introuvable pour preview: {src_path}")
        return None

    cached = preview_cache_path(cache_dir, src_path)

    if not cached.exists():
        try:
            run_make_preview_script(
                make_preview_script=make_preview_script,
                src_path=src_path,
                cache_dir=cache_dir,
                max_size=max_size,
                verbose=verbose,
            )
        except Exception as e:
            print(f"[WARN] Impossible de générer preview pour {src_path}: {e}")
            return None

    if not cached.exists():
        print(f"[WARN] Preview non généré après appel make_preview.py: {cached}")
        return None

    try:
        with Image.open(cached) as im:
            return np.array(im.convert("RGB"))
    except Exception as e:
        print(f"[WARN] Impossible de lire preview {cached}: {e}")
        return None


def create_image_mesh_on_film_back(T_wc, image_np, film_distance=12.0, film_width=12.0):
    """
    Crée un mesh triangulé coloré représentant l'image sur le fond de chambre.
    Cela évite l'effet 'micro images espacées' du point cloud.
    """
    h, w = image_np.shape[:2]
    aspect = w / h

    plane_w = film_width
    plane_h = film_width / aspect

    xs = np.linspace(-plane_w / 2.0, plane_w / 2.0, w)
    ys = np.linspace(-plane_h / 2.0, plane_h / 2.0, h)

    xv, yv = np.meshgrid(xs, ys)
    zv = np.full_like(xv, film_distance, dtype=np.float64)

    verts_cam = np.stack([xv, -yv, zv], axis=-1).reshape(-1, 3)
    colors = image_np.reshape(-1, 3).astype(np.float64) / 255.0

    triangles = []
    for y in range(h - 1):
        row0 = y * w
        row1 = (y + 1) * w
        for x in range(w - 1):
            i0 = row0 + x
            i1 = row0 + x + 1
            i2 = row1 + x
            i3 = row1 + x + 1
            triangles.append([i0, i2, i1])
            triangles.append([i1, i2, i3])

    triangles = np.asarray(triangles, dtype=np.int32)

    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3, 3]
    verts_world = (R_wc @ verts_cam.T).T + t_wc

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_world)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    return mesh


def visualize_colmap(frames, lidar, colmap_dir: Path, make_preview_script: Path,
                     show_images=True, frustum_scale=12.0, film_distance=12.0,
                     film_width=12.0, preview_size=1024, axis_size=3.0, verbose=False):
    geoms = [lidar]
    cache_dir = colmap_dir / "preview_cache"

    print("[INFO] Création des frustums, axes et images sur fond de chambre...")
    for i, frame in enumerate(frames):
        try:
            T_wc = build_T_wc_from_frame(frame)
        except Exception as e:
            print(f"[WARN] Frame #{i} ignorée: {e}")
            continue

        geoms.append(create_camera_frustum(T_wc, scale=frustum_scale, color=(1.0, 0.0, 0.0)))
        geoms.append(create_camera_axes(T_wc, axis_size=axis_size))

        if show_images:
            image_np = get_preview_image(
                frame,
                colmap_dir,
                cache_dir,
                make_preview_script=make_preview_script,
                max_size=preview_size,
                verbose=verbose,
            )
            if image_np is not None:
                try:
                    img_mesh = create_image_mesh_on_film_back(
                        T_wc,
                        image_np,
                        film_distance=film_distance,
                        film_width=film_width,
                    )
                    geoms.append(img_mesh)
                except Exception as e:
                    print(f"[WARN] Impossible d'ajouter l'image 3D pour {frame.get('file_path')}: {e}")

    print("[INFO] Lancement du viewer...")
    o3d.visualization.draw_geometries(
        geoms,
        window_name="COLMAP Viewer + Images on Film Back"
    )


def main():
    ap = argparse.ArgumentParser(description="Affiche les données COLMAP dans un viewer 3D avec images sur fond de chambre")
    ap.add_argument("--colmap-dir", required=True, help="Répertoire contenant transforms.json et le nuage de points")
    ap.add_argument("--make-preview-script", required=True, help="Chemin vers le script make_preview.py")
    ap.add_argument("--no-images", action="store_true", help="Désactive l'affichage des previews image en 3D")
    ap.add_argument("--frustum-scale", type=float, default=12.0, help="Taille du frustum caméra")
    ap.add_argument("--axis-size", type=float, default=3.0, help="Taille des axes caméra")
    ap.add_argument("--film-distance", type=float, default=12.0, help="Distance du fond de chambre")
    ap.add_argument("--film-width", type=float, default=12.0, help="Largeur du fond de chambre")
    ap.add_argument("--preview-size", type=int, default=1024, help="Taille max des previews en cache")
    ap.add_argument("--verbose", action="store_true", help="Affiche les logs détaillés")
    args = ap.parse_args()

    colmap_dir = Path(args.colmap_dir)
    make_preview_script = Path(args.make_preview_script)
    transforms_path = colmap_dir / "transforms.json"

    if not transforms_path.exists():
        raise FileNotFoundError(f"Fichier 'transforms.json' introuvable dans {transforms_path}")

    if not make_preview_script.exists():
        raise FileNotFoundError(f"Script make_preview.py introuvable: {make_preview_script}")

    print("[INFO] Chargement des données JSON...")
    data = load_json(transforms_path)
    frames = data.get("frames", [])
    if not frames:
        raise ValueError("Aucune caméra ou frame valide détectée dans transforms.json")

    lidar = load_lidar_ply(colmap_dir)
    visualize_colmap(
        frames,
        lidar,
        colmap_dir=colmap_dir,
        make_preview_script=make_preview_script,
        show_images=not args.no_images,
        frustum_scale=args.frustum_scale,
        film_distance=args.film_distance,
        film_width=args.film_width,
        preview_size=args.preview_size,
        axis_size=args.axis_size,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

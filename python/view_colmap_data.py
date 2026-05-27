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
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
except ImportError:
    print("Erreur: open3d requis. Installez avec : pip install open3d")
    raise

try:
    from PIL import Image
except ImportError:
    print("Erreur: pillow requis. Installez avec : pip install pillow")
    raise


def info(msg: str):
    print(f"[INFO] {msg}")


def warn(msg: str):
    print(f"[WARN] {msg}")


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


def create_camera_frustum(T_wc, scale=200.0, aspect=1.0, color=(1.0, 0.0, 0.0)):
    """
    Crée un frustum dont le fond de chambre est exactement cohérent
    avec le plan image affiché.

    - z = scale
    - largeur fond = scale
    - hauteur fond = scale / aspect
    """
    half_w = scale / 2.0
    half_h = (scale / aspect) / 2.0

    pts_cam = np.array([
        [0.0, 0.0, 0.0],
        [-half_w, -half_h, scale],
        [ half_w, -half_h, scale],
        [ half_w,  half_h, scale],
        [-half_w,  half_h, scale],
    ], dtype=np.float64)

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


def load_lidar_ply(colmap_dir: Path):
    candidate_paths = [
        colmap_dir / "points.ply",
        colmap_dir / "sparse_pc.ply",
    ]

    for path in candidate_paths:
        if path.exists():
            info(f"Chargement du nuage de points : {path}")
            pcd = o3d.io.read_point_cloud(str(path))
            if len(pcd.points) == 0:
                warn(f"Le nuage {path} est vide.")
            return pcd

    raise FileNotFoundError(
        f"Aucun fichier de nuage de points valide trouvé dans {colmap_dir}. "
        f"Recherchés : {[str(p) for p in candidate_paths]}"
    )


def apply_z_scale_to_geometry(geometry, z_scale: float):
    """
    Applique une échelle verticale sur Z à une géométrie Open3D.
    """
    if abs(z_scale - 1.0) < 1e-12:
        return geometry

    if isinstance(geometry, o3d.geometry.PointCloud):
        pts = np.asarray(geometry.points).copy()
        if pts.size > 0:
            pts[:, 2] *= z_scale
            geometry.points = o3d.utility.Vector3dVector(pts)
        return geometry

    if isinstance(geometry, o3d.geometry.LineSet):
        pts = np.asarray(geometry.points).copy()
        if pts.size > 0:
            pts[:, 2] *= z_scale
            geometry.points = o3d.utility.Vector3dVector(pts)
        return geometry

    if isinstance(geometry, o3d.geometry.TriangleMesh):
        verts = np.asarray(geometry.vertices).copy()
        if verts.size > 0:
            verts[:, 2] *= z_scale
            geometry.vertices = o3d.utility.Vector3dVector(verts)
            geometry.compute_vertex_normals()
        return geometry

    return geometry


def colorize_point_cloud_by_z(pcd: o3d.geometry.PointCloud):
    """
    Si le nuage est blanc/sans couleurs utiles, applique une coloration par altitude Z.
    """
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return pcd

    use_existing = False
    if pcd.has_colors():
        cols = np.asarray(pcd.colors)
        if cols.size > 0:
            # Si les couleurs ne sont pas quasi uniformes, on les garde.
            cstd = np.std(cols, axis=0).mean()
            use_existing = cstd > 1e-3

    if use_existing:
        info("Le nuage contient déjà des couleurs utiles, on les conserve.")
        return pcd

    z = pts[:, 2]
    zmin = np.min(z)
    zmax = np.max(z)

    if zmax <= zmin:
        cols = np.tile(np.array([[0.2, 0.7, 1.0]]), (len(pts), 1))
        pcd.colors = o3d.utility.Vector3dVector(cols)
        warn("Nuage plat en Z, application d'une couleur uniforme.")
        return pcd

    zn = (z - zmin) / (zmax - zmin)

    # petite rampe lisible: bleu -> cyan -> jaune -> rouge
    cols = np.zeros((len(zn), 3), dtype=np.float64)
    cols[:, 0] = np.clip(1.5 * zn - 0.5, 0.0, 1.0)
    cols[:, 1] = np.clip(1.5 - np.abs(2.0 * zn - 1.0) * 1.5, 0.0, 1.0)
    cols[:, 2] = np.clip(1.0 - 1.5 * zn, 0.0, 1.0)

    pcd.colors = o3d.utility.Vector3dVector(cols)
    info("Coloration du nuage appliquée selon l'altitude Z.")
    return pcd


def preview_cache_path(cache_dir: Path, image_path: Path):
    return cache_dir / f"{image_path.stem}.jpg"


def run_make_preview_script(make_preview_script: Path, src_path: Path, cache_dir: Path,
                            max_size=256, verbose=False):
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

    info(f"Exécution make_preview.py sur {src_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if verbose and result.stdout.strip():
        print(result.stdout.strip())
    if verbose and result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Échec de make_preview.py pour {src_path}\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )


def ensure_preview(frame, colmap_dir: Path, cache_dir: Path,
                   make_preview_script: Path, max_size=256, verbose=False):
    file_path = frame.get("file_path")
    if not file_path:
        warn("Frame sans file_path, impossible de générer le preview.")
        return None

    src_path = colmap_dir / file_path
    if not src_path.exists():
        warn(f"Image introuvable pour preview: {src_path}")
        return None

    cached = preview_cache_path(cache_dir, src_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if cached.exists():
        if verbose:
            info(f"Preview déjà présent: {cached.name}")
        return cached

    try:
        run_make_preview_script(
            make_preview_script=make_preview_script,
            src_path=src_path,
            cache_dir=cache_dir,
            max_size=max_size,
            verbose=verbose,
        )
    except Exception as e:
        warn(f"Impossible de générer preview pour {src_path}: {e}")
        return None

    if not cached.exists():
        warn(f"Preview non généré après appel make_preview.py: {cached}")
        return None

    if verbose:
        info(f"Preview généré: {cached.name}")
    return cached


def create_textured_image_quad(T_wc, scale=200.0, aspect=1.0):
    """
    Crée un quad exactement sur le fond de chambre du frustum.
    """
    half_w = scale / 2.0
    half_h = (scale / aspect) / 2.0

    verts_cam = np.array([
        [-half_w,  half_h, scale],
        [ half_w,  half_h, scale],
        [ half_w, -half_h, scale],
        [-half_w, -half_h, scale],
    ], dtype=np.float64)

    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)

    triangle_uvs = np.array([
        [0.0, 1.0], [1.0, 1.0], [1.0, 0.0],
        [0.0, 1.0], [1.0, 0.0], [0.0, 0.0],
    ], dtype=np.float64)

    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3, 3]
    verts_world = (R_wc @ verts_cam.T).T + t_wc

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_world)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    mesh.triangle_material_ids = o3d.utility.IntVector([0, 0])
    mesh.compute_vertex_normals()
    return mesh


def add_textured_quad_to_scene(vis, name, mesh, texture_path: Path):
    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.base_color = [1.0, 1.0, 1.0, 1.0]
    material.base_roughness = 1.0
    material.base_reflectance = 0.0
    material.base_metallic = 0.0

    tex = o3d.io.read_image(str(texture_path))
    tex_np = np.asarray(tex)
    if tex_np.size == 0:
        raise ValueError(f"Texture vide: {texture_path}")
    #info(f"Texture lue: {texture_path.name} shape={tex_np.shape}")

    material.albedo_img = tex
    vis.add_geometry(name, mesh, material)


def launch_o3d_visualizer(frames, lidar, colmap_dir: Path, make_preview_script: Path,
                          show_images=True, frustum_scale=200.0, film_distance=200.0,
                          image_size=300.0, preview_size=256, point_size=2.0,
                          z_scale=1.0, verbose=False):
    cache_dir = colmap_dir / "preview_cache"

    # préparation nuage : seul le nuage est affecté par z_scale,
    # en conservant fixe le Z minimal
    lidar = o3d.geometry.PointCloud(lidar)
    lidar = colorize_point_cloud_by_z(lidar)

    if abs(z_scale - 1.0) > 1e-12:
        pts = np.asarray(lidar.points).copy()
        if pts.size > 0:
            zmin = np.min(pts[:, 2])
            pts[:, 2] = zmin + z_scale * (pts[:, 2] - zmin)
            lidar.points = o3d.utility.Vector3dVector(pts)
            info(f"Échelle Z du nuage appliquée autour de zmin={zmin:.3f} avec facteur {z_scale}")
    else:
        info("z_scale=1.0, aucune modification verticale du nuage.")

    app = gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer("COLMAP Viewer + Textured Images", 1600, 1000)
    vis.show_settings = True
    vis.scene.set_background([0.05, 0.05, 0.05, 1.0])

    mat_pcd = rendering.MaterialRecord()
    mat_pcd.shader = "defaultUnlit"
    mat_pcd.point_size = point_size
    vis.add_geometry("lidar", lidar, mat_pcd)

    shown_images = 0

    for i, frame in enumerate(frames):
        try:
            T_wc = build_T_wc_from_frame(frame)
        except Exception as e:
            warn(f"Frame #{i} ignorée: {e}")
            continue

        aspect = 1.0
        texture_path = None

        if show_images:
            texture_path = ensure_preview(
                frame,
                colmap_dir,
                cache_dir,
                make_preview_script=make_preview_script,
                max_size=preview_size,
                verbose=verbose,
            )
            if texture_path is not None:
                try:
                    with Image.open(texture_path) as im:
                        w, h = im.size
                    if h > 0 and w > 0:
                        aspect = w / h
                    else:
                        warn(f"Texture invalide: {texture_path} ({w}x{h})")
                        texture_path = None
                except Exception as e:
                    warn(f"Impossible de lire la taille de texture pour {frame.get('file_path')}: {e}")
                    texture_path = None

        frustum = create_camera_frustum(
            T_wc,
            scale=frustum_scale,
            aspect=aspect,
            color=(1.0, 0.0, 0.0),
        )
        mat_line = rendering.MaterialRecord()
        mat_line.shader = "unlitLine"
        mat_line.line_width = 4.0
        vis.add_geometry(f"frustum_{i}", frustum, mat_line)

        if not show_images or texture_path is None:
            if show_images:
                warn(f"Pas de texture exploitable pour frame #{i}")
            continue

        try:
            quad = create_textured_image_quad(
                T_wc,
                scale=frustum_scale,
                aspect=aspect,
            )
            add_textured_quad_to_scene(vis, f"img_{i}", quad, texture_path)
            shown_images += 1
            if verbose:
                info(f"Image texturée ajoutée au fond de chambre: {frame.get('file_path')}")
        except Exception as e:
            warn(f"Impossible d'ajouter l'image texturée pour {frame.get('file_path')}: {e}")

    info(f"Nombre total d'images texturées ajoutées: {shown_images}")
    if shown_images == 0 and show_images:
        warn("Aucune image texturée n'a été ajoutée à la scène.")

    bounds = lidar.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = np.max(bounds.get_extent())
    eye = center + np.array([0.0, -2.0 * max(extent, 1.0), 1.0 * max(extent, 1.0)])
    up = [0.0, 0.0, 1.0]
    vis.setup_camera(60.0, center, eye, up)

    app.add_window(vis)
    app.run()


def main():
    script_dir = Path(__file__).resolve().parent
    default_make_preview = script_dir / "make_preview.py"

    ap = argparse.ArgumentParser(description="Affiche les données COLMAP avec images texturées au fond de chambre")
    ap.add_argument("--colmap-dir", required=True, help="Répertoire contenant transforms.json et le nuage de points")
    ap.add_argument(
        "--make-preview-script",
        default=str(default_make_preview),
        help="Chemin vers le script make_preview.py (défaut: même dossier que ce script)",
    )
    ap.add_argument("--no-images", action="store_true", help="Désactive l'affichage des images")
    ap.add_argument("--frustum-scale", type=float, default=200.0, help="Taille du frustum caméra")
    ap.add_argument("--film-distance", type=float, default=200.0, help="Distance du fond de chambre")
    ap.add_argument("--image-size", type=float, default=300.0, help="Largeur de l'image affichée")
    ap.add_argument("--preview-size", type=int, default=256, help="Taille max des previews en cache")
    ap.add_argument("--point-size", type=float, default=2.0, help="Taille d'affichage des points du nuage")
    ap.add_argument("--z-scale", type=float, default=1.0, help="Facteur d'échelle appliqué à Z pour l'affichage")
    ap.add_argument("--verbose", action="store_true", help="Affiche les logs détaillés")
    args = ap.parse_args()

    colmap_dir = Path(args.colmap_dir)
    make_preview_script = Path(args.make_preview_script)
    transforms_path = colmap_dir / "transforms.json"

    if not transforms_path.exists():
        raise FileNotFoundError(f"Fichier 'transforms.json' introuvable dans {transforms_path}")

    if not make_preview_script.exists():
        raise FileNotFoundError(f"Script make_preview.py introuvable: {make_preview_script}")

    info("Chargement des données JSON...")
    data = load_json(transforms_path)
    frames = data.get("frames", [])
    if not frames:
        raise ValueError("Aucune caméra ou frame valide détectée dans transforms.json")

    info(f"Nombre de frames chargées: {len(frames)}")
    lidar = load_lidar_ply(colmap_dir)

    launch_o3d_visualizer(
        frames,
        lidar,
        colmap_dir=colmap_dir,
        make_preview_script=make_preview_script,
        show_images=not args.no_images,
        frustum_scale=args.frustum_scale,
        film_distance=args.film_distance,
        image_size=args.image_size,
        preview_size=args.preview_size,
        point_size=args.point_size,
        z_scale=args.z_scale,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

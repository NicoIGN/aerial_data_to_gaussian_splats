#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import struct
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


def qvec_to_rotmat(qvec):
    qvec = np.asarray(qvec, dtype=np.float64)
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec

    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z,     2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z,     1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y,     2 * y * z + 2 * w * x,     1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def build_T_wc_from_colmap_image(colmap_image):
    qvec = np.asarray(colmap_image["qvec"], dtype=np.float64)
    tvec = np.asarray(colmap_image["tvec"], dtype=np.float64)

    R_cw = qvec_to_rotmat(qvec)
    R_wc = R_cw.T
    C = -R_wc @ tvec

    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = C
    return T_wc


def create_camera_frustum(T_wc, scale=1.0, aspect=1.0, color=(1.0, 0.0, 0.0), flip_z=False):
    half_w = scale / 2.0
    half_h = (scale / max(aspect, 1e-12)) / 2.0

    pts_cam = np.array([
        [0.0, 0.0, 0.0],
        [-half_w, -half_h, scale],
        [half_w, -half_h, scale],
        [half_w, half_h, scale],
        [-half_w, half_h, scale],
    ], dtype=np.float64)

    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3, 3]
    pts_world = (R_wc @ pts_cam.T).T + t_wc

    if flip_z:
        pts_world[:, 2] *= -1.0

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
    ]
    colors = np.tile(np.array(color, dtype=np.float64), (len(lines), 1))

    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(pts_world)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector(colors)
    return frustum


def create_textured_image_quad(T_wc, scale=1.0, aspect=1.0, flip_z=False):
    half_w = scale / 2.0
    half_h = (scale / max(aspect, 1e-12)) / 2.0

    verts_cam = np.array([
        [-half_w, half_h, scale],
        [half_w, half_h, scale],
        [half_w, -half_h, scale],
        [-half_w, -half_h, scale],
    ], dtype=np.float64)

    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)

    triangle_uvs = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0],
        [0.0, 0.0], [1.0, 1.0], [0.0, 1.0],
    ], dtype=np.float64)

    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3, 3]
    verts_world = (R_wc @ verts_cam.T).T + t_wc

    if flip_z:
        verts_world[:, 2] *= -1.0

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_world)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    mesh.triangle_material_ids = o3d.utility.IntVector([0, 0])
    mesh.compute_vertex_normals()
    return mesh


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


def colorize_point_cloud_by_z(pcd: o3d.geometry.PointCloud):
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return pcd

    use_existing = False
    if pcd.has_colors():
        cols = np.asarray(pcd.colors)
        if cols.size > 0:
            cstd = np.std(cols, axis=0).mean()
            use_existing = cstd > 1e-3

    if use_existing:
        return pcd

    z = pts[:, 2]
    zmin = np.min(z)
    zmax = np.max(z)

    if zmax <= zmin:
        cols = np.tile(np.array([[0.2, 0.7, 1.0]]), (len(pts), 1))
        pcd.colors = o3d.utility.Vector3dVector(cols)
        return pcd

    zn = (z - zmin) / (zmax - zmin)

    cols = np.zeros((len(zn), 3), dtype=np.float64)
    cols[:, 0] = np.clip(1.5 * zn - 0.5, 0.0, 1.0)
    cols[:, 1] = np.clip(1.5 - np.abs(2.0 * zn - 1.0) * 1.5, 0.0, 1.0)
    cols[:, 2] = np.clip(1.0 - 1.5 * zn, 0.0, 1.0)

    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def load_3dpoints(colmap_dir: Path):
    def _load_colmap_points3d_txt(path: Path):
        points = []
        colors = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 7:
                    continue

                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                r = int(parts[4])
                g = int(parts[5])
                b = int(parts[6])

                points.append([x, y, z])
                colors.append([r / 255.0, g / 255.0, b / 255.0])

        pcd = o3d.geometry.PointCloud()
        if points:
            pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
            pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
            info(f"{len(points)} points chargés depuis {path}")
        else:
            warn(f"Aucun point chargé depuis {path}")
        return pcd

    def _load_colmap_points3d_bin(path: Path):
        points = []
        colors = []

        with open(path, "rb") as f:
            num_points_data = f.read(8)
            if len(num_points_data) != 8:
                raise ValueError("Fichier points3D.bin invalide ou vide")

            num_points = struct.unpack("<Q", num_points_data)[0]

            for _ in range(num_points):
                point3d_id_data = f.read(8)
                xyz_data = f.read(24)
                rgb_data = f.read(3)
                error_data = f.read(8)
                track_len_data = f.read(8)

                if len(point3d_id_data) != 8 or len(xyz_data) != 24 or len(rgb_data) != 3 or len(error_data) != 8 or len(track_len_data) != 8:
                    raise ValueError("Fichier points3D.bin tronqué")

                _point3d_id = struct.unpack("<Q", point3d_id_data)[0]
                x, y, z = struct.unpack("<ddd", xyz_data)
                r, g, b = struct.unpack("<BBB", rgb_data)
                _error = struct.unpack("<d", error_data)[0]
                track_length = struct.unpack("<Q", track_len_data)[0]

                track_bytes = f.read(track_length * 8)
                if len(track_bytes) != track_length * 8:
                    raise ValueError("Fichier points3D.bin tronqué dans les tracks")

                points.append([x, y, z])
                colors.append([r / 255.0, g / 255.0, b / 255.0])

        pcd = o3d.geometry.PointCloud()
        if points:
            pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
            pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
            info(f"{len(points)} points chargés depuis {path}")
        else:
            warn(f"Aucun point chargé depuis {path}")
        return pcd

    candidate_paths = [
        colmap_dir / "colmap" / "sparse" / "0" / "points3D.bin",
        colmap_dir / "colmap" / "sparse" / "0_TXT" / "points3D.txt",
        colmap_dir / "sparse" / "0" / "points3D.bin",
        colmap_dir / "sparse" / "0_TXT" / "points3D.txt",
    ]

    for path in candidate_paths:
        if path.exists():
            info(f"Chargement des points 3D COLMAP : {path}")
            if path.suffix.lower() == ".bin":
                return _load_colmap_points3d_bin(path)
            elif path.suffix.lower() == ".txt":
                return _load_colmap_points3d_txt(path)

    raise FileNotFoundError(
        "Impossible de trouver points3D.bin ou points3D.txt "
        "dans colmap/sparse/0 ou colmap/sparse/0_TXT"
    )


def load_colmap_images(colmap_dir: Path):
    bin_candidates = [
        colmap_dir / "colmap" / "sparse" / "0" / "images.bin",
        colmap_dir / "sparse" / "0" / "images.bin",
    ]

    txt_candidates = [
        colmap_dir / "colmap" / "sparse" / "0_TXT" / "images.txt",
        colmap_dir / "sparse" / "0_TXT" / "images.txt",
    ]

    def _read_images_bin(path: Path):
        images = {}

        with open(path, "rb") as f:
            num_reg_images = struct.unpack("<Q", f.read(8))[0]

            for _ in range(num_reg_images):
                image_id = struct.unpack("<I", f.read(4))[0]
                qvec = struct.unpack("<dddd", f.read(32))
                tvec = struct.unpack("<ddd", f.read(24))
                camera_id = struct.unpack("<I", f.read(4))[0]

                name_bytes = bytearray()
                while True:
                    ch = f.read(1)
                    if ch == b"\x00":
                        break
                    if ch == b"":
                        raise ValueError("Fin de fichier inattendue dans images.bin")
                    name_bytes.extend(ch)
                name = name_bytes.decode("utf-8")

                num_points2D = struct.unpack("<Q", f.read(8))[0]
                f.read(num_points2D * 24)

                images[image_id] = {
                    "image_id": image_id,
                    "qvec": qvec,
                    "tvec": tvec,
                    "camera_id": camera_id,
                    "name": name,
                }

        info(f"{len(images)} poses caméra COLMAP chargées depuis {path}")
        return images

    def _read_images_txt(path: Path):
        images = {}

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if not line or line.startswith("#"):
                i += 1
                continue

            parts = line.split()
            if len(parts) < 10:
                i += 1
                continue

            image_id = int(parts[0])
            qvec = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
            tvec = [float(parts[5]), float(parts[6]), float(parts[7])]
            camera_id = int(parts[8])
            name = parts[9]

            images[image_id] = {
                "image_id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            }

            i += 2

        info(f"{len(images)} poses caméra COLMAP chargées depuis {path}")
        return images

    for path in bin_candidates:
        if path.exists():
            return _read_images_bin(path)

    for path in txt_candidates:
        if path.exists():
            return _read_images_txt(path)

    raise FileNotFoundError(
        "images.bin / images.txt introuvable dans colmap/sparse/0 ou sparse/0"
    )


def launch_o3d_visualizer(frames, lidar, colmap_images, colmap_dir: Path, make_preview_script: Path,
                          show_images=True, show_frustums=True,
                          frustum_scale=1.0, preview_size=256, point_size=2.0,
                          z_scale=1.0, flip_z=False, verbose=False):
    cache_dir = colmap_dir / "preview_cache"
    raw_lidar = o3d.geometry.PointCloud(lidar)

    app = gui.Application.instance
    app.initialize()

    window = app.create_window("Visualisation COLMAP", 1750, 1020)

    em = window.theme.font_size
    margin = 0.5 * em

    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    scene_widget.scene.set_background([0.08, 0.08, 0.08, 1.0])

    panel = gui.Vert(0.25 * em, gui.Margins(margin, margin, margin, margin))
    panel_width = int(24 * em)

    panel.add_child(gui.Label("Affichage"))

    pointcloud_checkbox = gui.Checkbox("Afficher le point cloud")
    pointcloud_checkbox.checked = True
    panel.add_child(pointcloud_checkbox)

    camera_checkbox = gui.Checkbox("Afficher les positions caméra")
    camera_checkbox.checked = show_frustums
    panel.add_child(camera_checkbox)

    image_checkbox = gui.Checkbox("Afficher les images")
    image_checkbox.checked = show_images
    panel.add_child(image_checkbox)

    flipz_checkbox = gui.Checkbox("Inverser axe Z")
    flipz_checkbox.checked = flip_z
    panel.add_child(flipz_checkbox)

    recenter_button = gui.Button("Recentrer sur bbox caméras")
    panel.add_child(recenter_button)

    zscale_label = gui.Label(f"Échelle Z : {z_scale:.2f}")
    panel.add_child(zscale_label)

    zscale_slider = gui.Slider(gui.Slider.DOUBLE)
    zscale_slider.set_limits(0.1, 10.0)
    zscale_slider.double_value = float(z_scale)
    panel.add_child(zscale_slider)

    pointsize_label = gui.Label(f"Taille des points : {point_size:.1f}")
    panel.add_child(pointsize_label)

    pointsize_slider = gui.Slider(gui.Slider.DOUBLE)
    pointsize_slider.set_limits(1.0, 10.0)
    pointsize_slider.double_value = float(point_size)
    panel.add_child(pointsize_slider)

    camera_scale_label = gui.Label(f"Taille des caméras : {frustum_scale:.3f}")
    panel.add_child(camera_scale_label)

    camera_scale_slider = gui.Slider(gui.Slider.DOUBLE)
    camera_scale_slider.set_limits(0.05, 10.0)
    camera_scale_slider.double_value = float(frustum_scale)
    panel.add_child(camera_scale_slider)

    panel.add_child(gui.Label("Repère COLMAP normalisé"))
    panel.add_child(gui.Label("Raccourci : Ctrl/Cmd + Q ou Esc pour quitter"))

    window.add_child(panel)
    window.add_child(scene_widget)

    def _on_layout(layout_context):
        content = window.content_rect
        panel.frame = gui.Rect(content.x, content.y, panel_width, content.height)
        scene_widget.frame = gui.Rect(
            content.x + panel_width,
            content.y,
            content.width - panel_width,
            content.height,
        )

    window.set_on_layout(_on_layout)

    def _collect_world_bounds_for_display(z_scale_value, flip_z_value):
        arrays = []

        lidar_pts = np.asarray(raw_lidar.points)
        if lidar_pts.size > 0:
            pts = lidar_pts.copy()
            if flip_z_value:
                pts[:, 2] *= -1.0
            if abs(z_scale_value - 1.0) > 1e-12:
                zmin = np.min(pts[:, 2])
                pts[:, 2] = zmin + z_scale_value * (pts[:, 2] - zmin)
            arrays.append(pts)

        cam_centers = []
        for frame in frames:
            colmap_im_id = frame.get("colmap_im_id")
            if colmap_im_id is None:
                continue
            colmap_image = colmap_images.get(int(colmap_im_id))
            if colmap_image is None:
                continue
            T_wc = build_T_wc_from_colmap_image(colmap_image)
            c = T_wc[:3, 3].copy()
            if flip_z_value:
                c[2] *= -1.0
            cam_centers.append(c)

        if cam_centers:
            arrays.append(np.asarray(cam_centers, dtype=np.float64))

        if not arrays:
            return None, None, None

        all_pts = np.vstack(arrays)
        pmin = all_pts.min(axis=0)
        pmax = all_pts.max(axis=0)
        center = 0.5 * (pmin + pmax)
        extent = pmax - pmin
        max_extent = float(np.max(extent))

        if max_extent < 1e-12:
            scale = 1.0
        else:
            scale = 10.0 / max_extent

        return center, scale, extent

    initial_center, initial_scale, _ = _collect_world_bounds_for_display(z_scale, flip_z)
    if initial_center is None:
        initial_center = np.zeros(3, dtype=np.float64)
        initial_scale = 1.0

    state = {
        "show_pointcloud": True,
        "show_cameras": bool(show_frustums),
        "show_images": bool(show_images),
        "z_scale": float(z_scale),
        "point_size": float(point_size),
        "frustum_scale": float(frustum_scale),
        "flip_z": bool(flip_z),
        "display_center": np.asarray(initial_center, dtype=np.float64),
        "display_scale": float(initial_scale),
        "camera_names": [],
        "image_names": [],
        "view_initialized": False,
    }

    def _compute_camera_centers_world():
        centers = []

        for frame in frames:
            colmap_im_id = frame.get("colmap_im_id")
            if colmap_im_id is None:
                continue

            colmap_image = colmap_images.get(int(colmap_im_id))
            if colmap_image is None:
                continue

            T_wc = build_T_wc_from_colmap_image(colmap_image)
            c = T_wc[:3, 3].copy()

            if state["flip_z"]:
                c[2] *= -1.0

            centers.append(c)

        if not centers:
            return np.zeros((0, 3), dtype=np.float64)

        return np.asarray(centers, dtype=np.float64)

    def _world_to_display(points):
        pts = np.asarray(points, dtype=np.float64).copy()
        pts = (pts - state["display_center"]) * state["display_scale"]
        return pts

    def _recompute_display_transform():
        center, scale, _ = _collect_world_bounds_for_display(state["z_scale"], state["flip_z"])
        if center is None:
            return
        state["display_center"] = np.asarray(center, dtype=np.float64)
        state["display_scale"] = float(scale)

    def _build_display_lidar():
        lidar_local = o3d.geometry.PointCloud(raw_lidar)
        lidar_local = colorize_point_cloud_by_z(lidar_local)

        pts = np.asarray(lidar_local.points).copy()
        if pts.size > 0:
            if state["flip_z"]:
                pts[:, 2] *= -1.0

            if abs(state["z_scale"] - 1.0) > 1e-12:
                zmin = np.min(pts[:, 2])
                pts[:, 2] = zmin + state["z_scale"] * (pts[:, 2] - zmin)

            pts = _world_to_display(pts)
            lidar_local.points = o3d.utility.Vector3dVector(pts)

        return lidar_local

    def _clear_dynamic_geometry():
        names = ["lidar"] + state["camera_names"] + state["image_names"]
        for name in names:
            try:
                scene_widget.scene.remove_geometry(name)
            except Exception:
                pass
        state["camera_names"] = []
        state["image_names"] = []

    def _setup_camera_with_bounds(bounds):
        center = bounds.get_center()
        scene_widget.setup_camera(60.0, bounds, center)
        try:
            extent = np.linalg.norm(bounds.get_extent())
            near = max(extent * 0.0005, 0.001)
            far = max(extent * 4.0, 10.0)
            scene_widget.scene.camera.set_projection(
                60.0,
                scene_widget.frame.width / max(scene_widget.frame.height, 1),
                near,
                far,
                rendering.Camera.FovType.Vertical
            )
        except Exception:
            pass

    def _rebuild_scene(reset_camera=False):
        _clear_dynamic_geometry()

        _recompute_display_transform()

        lidar_local = _build_display_lidar()

        mat_pcd = rendering.MaterialRecord()
        mat_pcd.shader = "defaultUnlit"
        mat_pcd.point_size = state["point_size"]

        bounds_candidates = []
        camera_centers_for_view = []

        if state["show_pointcloud"] and len(lidar_local.points) > 0:
            scene_widget.scene.add_geometry("lidar", lidar_local, mat_pcd)
            bounds_candidates.append(np.asarray(lidar_local.points))
        elif len(lidar_local.points) == 0:
            warn("Aucun point 3D à afficher.")

        camera_scene_extent = 10.0
        base_camera_size = 0.03 * camera_scene_extent

        shown_images = 0
        shown_cameras = 0
        missing_ids = 0

        for i, frame in enumerate(frames):
            colmap_im_id = frame.get("colmap_im_id")
            if colmap_im_id is None:
                warn(f"Frame #{i} sans colmap_im_id, ignorée pour la géométrie caméra.")
                continue

            colmap_image = colmap_images.get(int(colmap_im_id))
            if colmap_image is None:
                warn(f"Pose COLMAP introuvable pour colmap_im_id={colmap_im_id}")
                missing_ids += 1
                continue

            T_wc = build_T_wc_from_colmap_image(colmap_image)

            camera_center = T_wc[:3, 3].copy()
            if state["flip_z"]:
                camera_center[2] *= -1.0
            camera_center = _world_to_display(camera_center.reshape(1, 3))[0]

            bounds_candidates.append(camera_center.reshape(1, 3))
            camera_centers_for_view.append(camera_center.reshape(1, 3))

            aspect = 1.0
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
                        texture_path = None
                except Exception as e:
                    warn(f"Impossible de lire la taille de texture pour {frame.get('file_path')}: {e}")
                    texture_path = None

            current_camera_scale = state["frustum_scale"] * base_camera_size / max(state["display_scale"], 1e-12)

            if state["show_cameras"]:
                try:
                    camera_name = f"camera_{i}"
                    frustum = create_camera_frustum(
                        T_wc,
                        scale=current_camera_scale,
                        aspect=aspect,
                        color=(1.0, 0.0, 0.0),
                        flip_z=state["flip_z"],
                    )

                    pts = np.asarray(frustum.points).copy()
                    pts = _world_to_display(pts)
                    frustum.points = o3d.utility.Vector3dVector(pts)

                    mat_line = rendering.MaterialRecord()
                    mat_line.shader = "unlitLine"
                    mat_line.line_width = 2.0
                    scene_widget.scene.add_geometry(camera_name, frustum, mat_line)
                    state["camera_names"].append(camera_name)
                    shown_cameras += 1
                except Exception as e:
                    warn(f"Impossible d'ajouter la position caméra pour {frame.get('file_path', f'frame #{i}')} : {e}")

            if not state["show_images"] or texture_path is None:
                continue

            try:
                quad = create_textured_image_quad(
                    T_wc,
                    scale=current_camera_scale,
                    aspect=aspect,
                    flip_z=state["flip_z"],
                )

                verts = np.asarray(quad.vertices).copy()
                verts = _world_to_display(verts)
                quad.vertices = o3d.utility.Vector3dVector(verts)
                quad.compute_vertex_normals()

                material = rendering.MaterialRecord()
                material.shader = "defaultUnlit"
                material.base_color = [1.0, 1.0, 1.0, 1.0]
                material.albedo_img = o3d.io.read_image(str(texture_path))

                image_name = f"img_{i}"
                scene_widget.scene.add_geometry(image_name, quad, material)
                state["image_names"].append(image_name)
                shown_images += 1
            except Exception as e:
                warn(f"Impossible d'ajouter l'image texturée pour {frame.get('file_path')}: {e}")

        info(f"Nombre total de positions caméra ajoutées: {shown_cameras}")
        info(f"Nombre total d'images texturées ajoutées: {shown_images}")
        if missing_ids > 0:
            warn(f"{missing_ids} frames ignorées faute de pose COLMAP.")

        if reset_camera or not state["view_initialized"]:
            target_arrays = camera_centers_for_view if camera_centers_for_view else bounds_candidates

            if target_arrays:
                all_pts = np.vstack(target_arrays)
                pcd_bounds = o3d.geometry.PointCloud()
                pcd_bounds.points = o3d.utility.Vector3dVector(all_pts)
                bounds = pcd_bounds.get_axis_aligned_bounding_box()

                extent = bounds.get_extent()
                pad = np.maximum(extent * 0.1, 1e-3)
                bounds = o3d.geometry.AxisAlignedBoundingBox(
                    bounds.min_bound - pad,
                    bounds.max_bound + pad
                )

                _setup_camera_with_bounds(bounds)
                state["view_initialized"] = True

    def _refresh_scene():
        _rebuild_scene(reset_camera=False)

    def _on_toggle_pointcloud(checked):
        state["show_pointcloud"] = bool(checked)
        _refresh_scene()

    def _on_toggle_cameras(checked):
        state["show_cameras"] = bool(checked)
        _refresh_scene()

    def _on_toggle_images(checked):
        state["show_images"] = bool(checked)
        _refresh_scene()

    def _on_flip_z(checked):
        state["flip_z"] = bool(checked)
        _refresh_scene()

    def _on_recenter():
        centers = []
        for frame in frames:
            colmap_im_id = frame.get("colmap_im_id")
            if colmap_im_id is None:
                continue
            colmap_image = colmap_images.get(int(colmap_im_id))
            if colmap_image is None:
                continue
            T_wc = build_T_wc_from_colmap_image(colmap_image)
            c = T_wc[:3, 3].copy()
            if state["flip_z"]:
                c[2] *= -1.0
            centers.append(_world_to_display(c.reshape(1, 3))[0])

        if not centers:
            warn("Impossible de recentrer: aucune position caméra disponible.")
            return

        centers = np.asarray(centers, dtype=np.float64)
        cmin = centers.min(axis=0)
        cmax = centers.max(axis=0)
        bounds = o3d.geometry.AxisAlignedBoundingBox(cmin, cmax)

        extent = bounds.get_extent()
        pad = np.maximum(extent * 0.1, 1e-3)
        bounds = o3d.geometry.AxisAlignedBoundingBox(
            bounds.min_bound - pad,
            bounds.max_bound + pad
        )

        _setup_camera_with_bounds(bounds)
        window.post_redraw()

    def _on_zscale_changed(value):
        state["z_scale"] = float(value)
        zscale_label.text = f"Échelle Z : {state['z_scale']:.2f}"
        _refresh_scene()

    def _on_pointsize_changed(value):
        state["point_size"] = float(value)
        pointsize_label.text = f"Taille des points : {state['point_size']:.1f}"
        _refresh_scene()

    def _on_camera_scale_changed(value):
        state["frustum_scale"] = float(value)
        camera_scale_label.text = f"Taille des caméras : {state['frustum_scale']:.3f}"
        _refresh_scene()

    def _on_key(event):
        if event.type == gui.KeyEvent.DOWN:
            if event.key == gui.KeyName.Q and (
                event.is_modifier_down(gui.KeyModifier.CTRL)
                or event.is_modifier_down(gui.KeyModifier.META)
            ):
                gui.Application.instance.quit()
                return True

            if event.key == gui.KeyName.ESCAPE:
                gui.Application.instance.quit()
                return True

        return False

    scene_widget.set_on_key(_on_key)
    try:
        window.set_on_key(_on_key)
    except Exception:
        try:
            window.set_on_key_event(_on_key)
        except Exception:
            warn("Impossible d'attacher le raccourci clavier au niveau fenêtre.")

    pointcloud_checkbox.set_on_checked(_on_toggle_pointcloud)
    camera_checkbox.set_on_checked(_on_toggle_cameras)
    image_checkbox.set_on_checked(_on_toggle_images)
    flipz_checkbox.set_on_checked(_on_flip_z)
    recenter_button.set_on_clicked(_on_recenter)
    zscale_slider.set_on_value_changed(_on_zscale_changed)
    pointsize_slider.set_on_value_changed(_on_pointsize_changed)
    camera_scale_slider.set_on_value_changed(_on_camera_scale_changed)

    _rebuild_scene(reset_camera=True)
    app.run()


def main():
    script_dir = Path(__file__).resolve().parent
    default_make_preview = script_dir / "make_preview.py"

    ap = argparse.ArgumentParser(description="Affiche les points 3D et caméras COLMAP dans un repère normalisé")
    ap.add_argument("--colmap-dir", required=True, help="Répertoire racine contenant transforms.json et colmap/")
    ap.add_argument(
        "--make-preview-script",
        default=str(default_make_preview),
        help="Chemin vers le script make_preview.py",
    )
    ap.add_argument("--no-images", action="store_true", help="Désactive l'affichage des images")
    ap.add_argument("--no-frustums", action="store_true", help="Désactive l'affichage des positions caméra")
    ap.add_argument("--frustum-scale", type=float, default=1.0, help="Taille relative initiale des caméras")
    ap.add_argument("--preview-size", type=int, default=256, help="Taille max des previews en cache")
    ap.add_argument("--point-size", type=float, default=2.0, help="Taille d'affichage des points du nuage")
    ap.add_argument("--z-scale", type=float, default=1.0, help="Facteur d'échelle appliqué à Z pour les points 3D")
    ap.add_argument("--flip-z", action="store_true", help="Inverse l'axe Z au démarrage")
    ap.add_argument("--verbose", action="store_true", help="Affiche les logs détaillés")
    args = ap.parse_args()

    colmap_dir = Path(args.colmap_dir)
    make_preview_script = Path(args.make_preview_script)
    transforms_path = colmap_dir / "transforms.json"

    if not transforms_path.exists():
        raise FileNotFoundError(f"Fichier 'transforms.json' introuvable: {transforms_path}")

    if not make_preview_script.exists():
        raise FileNotFoundError(f"Script make_preview.py introuvable: {make_preview_script}")

    data = load_json(transforms_path)
    frames = data.get("frames", [])
    if not frames:
        raise ValueError("Aucune frame valide détectée dans transforms.json")

    info(f"Nombre de frames chargées: {len(frames)}")

    lidar = load_3dpoints(colmap_dir)
    colmap_images = load_colmap_images(colmap_dir)

    launch_o3d_visualizer(
        frames=frames,
        lidar=lidar,
        colmap_images=colmap_images,
        colmap_dir=colmap_dir,
        make_preview_script=make_preview_script,
        show_images=not args.no_images,
        show_frustums=not args.no_frustums,
        frustum_scale=args.frustum_scale,
        preview_size=args.preview_size,
        point_size=args.point_size,
        z_scale=args.z_scale,
        flip_z=args.flip_z,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

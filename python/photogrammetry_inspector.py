#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import struct
import sys
from pathlib import Path
import subprocess

import numpy as np

try:
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
except ImportError:
    print("Erreur: open3d requis. Installez avec : pip install open3d")
    raise

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Erreur: pillow requis. Installez avec : pip install pillow")
    raise


def info(msg: str):
    print(f"[INFO] {msg}")


def warn(msg: str):
    print(f"[WARN] {msg}")


def dbg(msg: str, verbose: bool):
    if verbose:
        print(f"[DEBUG] {msg}")


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


def create_camera_frustum(T_wc, depth, base_width, aspect=1.0, color=(1.0, 0.0, 0.0)):
    half_w = base_width / 2.0
    half_h = half_w / max(aspect, 1e-12)

    pts_cam = np.array([
        [0.0, 0.0, 0.0],
        [-half_w, -half_h, depth],
        [ half_w, -half_h, depth],
        [ half_w,  half_h, depth],
        [-half_w,  half_h, depth],
    ], dtype=np.float64)

    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3, 3]
    pts_world = (R_wc @ pts_cam.T).T + t_wc

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


def create_textured_image_quad(T_wc, depth, base_width, aspect=1.0):
    half_w = base_width / 2.0
    half_h = half_w / max(aspect, 1e-12)

    verts_cam = np.array([
        [-half_w,  half_h, depth],
        [ half_w,  half_h, depth],
        [ half_w, -half_h, depth],
        [-half_w, -half_h, depth],
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

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_world)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    mesh.triangle_material_ids = o3d.utility.IntVector([0, 0])
    mesh.compute_vertex_normals()
    return mesh


def colorize_point_cloud_by_z(pcd: o3d.geometry.PointCloud):
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return pcd

    if pcd.has_colors():
        cols = np.asarray(pcd.colors)
        if cols.size > 0 and np.std(cols, axis=0).mean() > 1e-3:
            return pcd

    z = pts[:, 2]
    zmin, zmax = np.min(z), np.max(z)

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


def _setup_camera_with_bounds(self, bounds):
    center = bounds.get_center()
    extent = bounds.get_extent()
    radius = 0.5 * float(np.linalg.norm(extent))

    if radius < 1e-6:
        radius = 1.0

    eye = center + np.array([0.0, -2.5 * radius, 1.2 * radius], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    self.scene_widget.look_at(center, eye, up)

    try:
        fov_deg = 60.0
        near = max(radius * 0.01, 0.001)
        far = max(radius * 20.0, 10.0)

        self.scene_widget.scene.camera.set_projection(
            fov_deg,
            self.scene_widget.frame.width / max(self.scene_widget.frame.height, 1),
            near,
            far,
            rendering.Camera.FovType.Vertical
        )
    except Exception:
        pass


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


def ensure_preview(src_path: Path, cache_dir: Path, make_preview_script: Path,
                   max_size=256, verbose=False):
    if not src_path.exists():
        warn(f"Image source introuvable pour preview: {src_path}")
        return None

    cached = preview_cache_path(cache_dir, src_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if cached.exists():
        if verbose:
            info(f"Preview déjà présent: {cached}")
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

    if cached.exists():
        if verbose:
            info(f"Preview généré: {cached}")
        return cached

    warn(f"Preview non généré après appel make_preview.py: {cached}")
    return None


def load_3dpoints(colmap_dir: Path):
    def _load_txt(path: Path):
        ids = []
        points = []
        colors = []
        observations = {}

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 8:
                    continue

                pid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                r = int(parts[4])
                g = int(parts[5])
                b = int(parts[6])

                ids.append(pid)
                points.append([x, y, z])
                colors.append([r / 255.0, g / 255.0, b / 255.0])

                image_ids = []
                track_parts = parts[8:]

                for i in range(0, len(track_parts), 2):
                    if i + 1 >= len(track_parts):
                        break
                    image_id = int(track_parts[i])
                    image_ids.append(image_id)

                observations[pid] = {
                    "image_ids": image_ids
                }

        pcd = o3d.geometry.PointCloud()

        if points:
            pcd.points = o3d.utility.Vector3dVector(
                np.asarray(points, dtype=np.float64)
            )
            pcd.colors = o3d.utility.Vector3dVector(
                np.asarray(colors, dtype=np.float64)
            )

        return {
            "pcd": pcd,
            "point_ids": np.asarray(ids, dtype=np.int64),
            "xyz": (
                np.asarray(points, dtype=np.float64)
                if points
                else np.zeros((0, 3), dtype=np.float64)
            ),
            "observations": observations,
        }

    def _load_bin(path: Path):
        ids = []
        points = []
        colors = []
        observations = {}

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

                if (
                    len(point3d_id_data) != 8
                    or len(xyz_data) != 24
                    or len(rgb_data) != 3
                    or len(error_data) != 8
                    or len(track_len_data) != 8
                ):
                    raise ValueError("Fichier points3D.bin tronqué")

                point3d_id = struct.unpack("<Q", point3d_id_data)[0]
                x, y, z = struct.unpack("<ddd", xyz_data)
                r, g, b = struct.unpack("<BBB", rgb_data)
                _error = struct.unpack("<d", error_data)[0]
                track_length = struct.unpack("<Q", track_len_data)[0]

                image_ids = []

                for _ in range(track_length):
                    image_id = struct.unpack("<I", f.read(4))[0]
                    _point2d_idx = struct.unpack("<I", f.read(4))[0]
                    image_ids.append(image_id)

                observations[point3d_id] = {
                    "image_ids": image_ids
                }

                ids.append(point3d_id)
                points.append([x, y, z])
                colors.append([r / 255.0, g / 255.0, b / 255.0])

        pcd = o3d.geometry.PointCloud()

        if points:
            pcd.points = o3d.utility.Vector3dVector(
                np.asarray(points, dtype=np.float64)
            )
            pcd.colors = o3d.utility.Vector3dVector(
                np.asarray(colors, dtype=np.float64)
            )

        return {
            "pcd": pcd,
            "point_ids": np.asarray(ids, dtype=np.int64),
            "xyz": (
                np.asarray(points, dtype=np.float64)
                if points
                else np.zeros((0, 3), dtype=np.float64)
            ),
            "observations": observations,
        }

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
                return _load_bin(path)

            return _load_txt(path)

    raise FileNotFoundError(
        "Impossible de trouver points3D.bin ou points3D.txt"
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

        return images

    for path in bin_candidates:
        if path.exists():
            images = _read_images_bin(path)
            info(f"{len(images)} poses caméra COLMAP chargées depuis {path}")
            return images

    for path in txt_candidates:
        if path.exists():
            images = _read_images_txt(path)
            info(f"{len(images)} poses caméra COLMAP chargées depuis {path}")
            return images

    raise FileNotFoundError("images.bin / images.txt introuvable")


def load_colmap_cameras(colmap_dir: Path):
    bin_candidates = [
        colmap_dir / "colmap" / "sparse" / "0" / "cameras.bin",
        colmap_dir / "sparse" / "0" / "cameras.bin",
    ]
    txt_candidates = [
        colmap_dir / "colmap" / "sparse" / "0_TXT" / "cameras.txt",
        colmap_dir / "sparse" / "0_TXT" / "cameras.txt",
    ]

    def _read_txt(path: Path):
        cams = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cam_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(v) for v in parts[4:]]
                cams[cam_id] = {
                    "camera_id": cam_id,
                    "model": model,
                    "width": width,
                    "height": height,
                    "params": params,
                }
        return cams

    def _read_bin(path: Path):
        cams = {}
        with open(path, "rb") as f:
            num_cameras = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_cameras):
                camera_id = struct.unpack("<I", f.read(4))[0]
                model_id = struct.unpack("<i", f.read(4))[0]
                width = struct.unpack("<Q", f.read(8))[0]
                height = struct.unpack("<Q", f.read(8))[0]

                model_num_params = {
                    0: 3,
                    1: 4,
                    2: 4,
                    3: 5,
                    4: 8,
                    5: 8,
                    6: 12,
                    7: 5,
                    8: 4,
                    9: 5,
                    10: 12,
                }
                nparams = model_num_params[model_id]
                params = list(struct.unpack("<" + "d" * nparams, f.read(8 * nparams)))

                model_names = {
                    0: "SIMPLE_PINHOLE",
                    1: "PINHOLE",
                    2: "SIMPLE_RADIAL",
                    3: "RADIAL",
                    4: "OPENCV",
                    5: "OPENCV_FISHEYE",
                    6: "FULL_OPENCV",
                    7: "FOV",
                    8: "SIMPLE_RADIAL_FISHEYE",
                    9: "RADIAL_FISHEYE",
                    10: "THIN_PRISM_FISHEYE",
                }

                cams[camera_id] = {
                    "camera_id": camera_id,
                    "model": model_names.get(model_id, f"MODEL_{model_id}"),
                    "width": int(width),
                    "height": int(height),
                    "params": params,
                }
        return cams

    for path in bin_candidates:
        if path.exists():
            cams = _read_bin(path)
            info(f"{len(cams)} caméras COLMAP chargées depuis {path}")
            return cams

    for path in txt_candidates:
        if path.exists():
            cams = _read_txt(path)
            info(f"{len(cams)} caméras COLMAP chargées depuis {path}")
            return cams

    raise FileNotFoundError("cameras.bin / cameras.txt introuvable")


def get_intrinsics_from_camera(cam):
    model = cam["model"]
    p = cam["params"]

    if model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
        dist = {"model": model, "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0}

    elif model == "SIMPLE_PINHOLE":
        f, cx, cy = p[:3]
        fx = fy = f
        dist = {"model": model, "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0}

    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p[:8]
        dist = {"model": model, "k1": k1, "k2": k2, "p1": p1, "p2": p2}

    else:
        raise NotImplementedError(
            f"Modèle caméra non supporté pour projection simple: {model}"
        )

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    return K, cam["width"], cam["height"], dist


def project_world_point(colmap_image, cam, xyz, z_positive=True):
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)

    qvec = np.asarray(colmap_image["qvec"], dtype=np.float64)
    tvec = np.asarray(colmap_image["tvec"], dtype=np.float64)
    R_cw = qvec_to_rotmat(qvec)

    Xc = R_cw @ xyz + tvec
    z_raw = float(Xc[2])

    if z_positive:
        if z_raw <= 1e-9:
            return None
        z = z_raw
    else:
        if z_raw >= -1e-9:
            return None
        z = -z_raw

    K, width, height, dist = get_intrinsics_from_camera(cam)

    x = float(Xc[0] / z)
    y = float(Xc[1] / z)

    model = dist["model"]

    if model in ("PINHOLE", "SIMPLE_PINHOLE"):
        xd = x
        yd = y

    elif model == "OPENCV":
        k1 = dist["k1"]
        k2 = dist["k2"]
        p1 = dist["p1"]
        p2 = dist["p2"]

        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2

        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    else:
        raise NotImplementedError(
            f"Modèle caméra non supporté pour projection: {model}"
        )

    u = K[0, 0] * xd + K[0, 2]
    v = K[1, 1] * yd + K[1, 2]

    inside = (0.0 <= u < width) and (0.0 <= v < height)

    return {
        "uv": np.array([u, v], dtype=np.float64),
        "depth": z,
        "inside": inside,
        "width": width,
        "height": height,
    }


def pil_to_o3d_image(pil_image: Image.Image):
    arr = np.asarray(pil_image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    arr = np.ascontiguousarray(arr.astype(np.uint8))
    return o3d.geometry.Image(arr)


def make_sphere(center, radius=1.0, color=(1.0, 1.0, 0.0)):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    mesh.translate(np.asarray(center, dtype=np.float64))
    return mesh


def resolve_image_path(images_dir: Path,
                       colmap_name: str,
                       preview_cache_dir: Path = None,
                       make_preview_script: Path = None,
                       preview_max_size: int = 256,
                       verbose=False):
    direct = images_dir / colmap_name
    stem = Path(colmap_name).stem

    source_path = None

    dbg(f"Recherche image directe: {direct}", verbose)
    if direct.exists():
        source_path = direct
    else:
        candidates = list(images_dir.glob(f"{stem}.*"))
        dbg(f"Recherche par stem '{stem}': {len(candidates)} candidat(s)", verbose)
        if candidates:
            source_path = candidates[0]

    if source_path is None:
        return None

    if preview_cache_dir is not None and make_preview_script is not None:
        cached = ensure_preview(
            src_path=source_path,
            cache_dir=preview_cache_dir,
            make_preview_script=make_preview_script,
            max_size=preview_max_size,
            verbose=verbose,
        )
        if cached is not None and cached.exists():
            return cached

    return source_path


def draw_zoomed_projections_on_image(image_path: Path, uv_fullres, full_w, full_h,
                                     target_size=420, padding_px=30, marker_radius=5):
    uv_fullres = np.asarray(uv_fullres, dtype=np.float64).reshape(-1, 2)

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        pw, ph = im.size

        sx = pw / max(full_w, 1)
        sy = ph / max(full_h, 1)

        uv_img = uv_fullres.copy()
        uv_img[:, 0] *= sx
        uv_img[:, 1] *= sy

        umin = float(np.min(uv_img[:, 0]))
        umax = float(np.max(uv_img[:, 0]))
        vmin = float(np.min(uv_img[:, 1]))
        vmax = float(np.max(uv_img[:, 1]))

        if len(uv_img) == 1:
            extra = 80
            umin -= extra
            umax += extra
            vmin -= extra
            vmax += extra

        left = max(int(np.floor(umin - padding_px)), 0)
        top = max(int(np.floor(vmin - padding_px)), 0)
        right = min(int(np.ceil(umax + padding_px)), pw)
        bottom = min(int(np.ceil(vmax + padding_px)), ph)

        if right <= left:
            right = min(left + 1, pw)
        if bottom <= top:
            bottom = min(top + 1, ph)

        crop = im.crop((left, top, right, bottom))
        crop_w, crop_h = crop.size

        scale = min(target_size / max(crop_w, 1), target_size / max(crop_h, 1))
        out_w = max(1, int(round(crop_w * scale)))
        out_h = max(1, int(round(crop_h * scale)))
        crop = crop.resize((out_w, out_h), Image.Resampling.LANCZOS)

        draw = ImageDraw.Draw(crop)
        for uv in uv_img:
            u = (uv[0] - left) * scale
            v = (uv[1] - top) * scale
            r = marker_radius
            draw.ellipse((u - r, v - r, u + r, v + r), outline=(255, 0, 0), width=2)
            draw.line((u - 2 * r, v, u + 2 * r, v), fill=(255, 0, 0), width=2)
            draw.line((u, v - 2 * r, u, v + 2 * r), fill=(255, 0, 0), width=2)

        return crop.copy()


class InspectorApp:
    def __init__(self, pointcloud_data, colmap_images, colmap_cameras,
                 colmap_dir: Path, point_size=2.0, frustum_scale=None, z_positive=True, verbose=False):

        self.pointcloud_data = pointcloud_data
        self.raw_pointcloud = o3d.geometry.PointCloud(pointcloud_data["pcd"])
        self.points_xyz = pointcloud_data["xyz"]
        self.point_ids = pointcloud_data["point_ids"]
        self.point_observations = pointcloud_data.get("observations", {})
        self.z_positive = bool(z_positive)
        
        self.point_id_to_global_index = {
            int(pid): idx for idx, pid in enumerate(self.point_ids)
        }

        tracked_ids = {
            int(pid)
            for pid, obs in self.point_observations.items()
            if obs is not None and len(obs.get("image_ids", [])) > 0
        }

        self.tracked_mask = np.array(
            [int(pid) in tracked_ids for pid in self.point_ids],
            dtype=bool
        )

        self.tracked_points_xyz = self.points_xyz[self.tracked_mask]
        self.tracked_point_ids = self.point_ids[self.tracked_mask]

        if self.raw_pointcloud.has_colors() and len(self.raw_pointcloud.points) == len(self.points_xyz):
            raw_colors = np.asarray(self.raw_pointcloud.colors)
            self.raw_point_colors = raw_colors
            self.tracked_point_colors = raw_colors[self.tracked_mask]
        else:
            self.raw_point_colors = None
            self.tracked_point_colors = None

        info(
            f"Points trackés: {int(np.count_nonzero(self.tracked_mask))} / {len(self.point_ids)}"
        )

        self.colmap_images = colmap_images
        self.colmap_cameras = colmap_cameras
        self.colmap_dir = colmap_dir
        self.images_dir = colmap_dir / "images"
        self.verbose = bool(verbose)

        self.selected_point_index = None
        self.selected_point_xyz = None
        self.selected_sphere_name = "selected_point_marker"

        self._camera_geom_names = []
        self._camera_image_geom_names = []

        self.preview_cache_dir = self.colmap_dir / "preview_cache"
        self.make_preview_script = Path(__file__).resolve().parent / "make_preview.py"
        self.preview_max_size = 256

        self.frustum_depth, self.frustum_basewidth = self._compute_camera_scale_defaults()
        self.camera_scale_min = 0.25
        self.camera_scale_max = 4.0

        if frustum_scale is None:
            frustum_scale = 1.0

        self.app = gui.Application.instance
        self.app.initialize()

        self.window = self.app.create_window(
            "Inspection photogrammétrique",
            1900,
            1080
        )

        self.em = self.window.theme.font_size
        self.margin = 0.5 * self.em

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(
            self.window.renderer
        )
        self.scene_widget.scene.set_background(
            [0.08, 0.08, 0.08, 1.0]
        )

        self.left_panel = gui.Vert(
            0.25 * self.em,
            gui.Margins(
                self.margin,
                self.margin,
                self.margin,
                self.margin
            )
        )

        self.right_panel_background = gui.Vert(
            0.25 * self.em,
            gui.Margins(
                self.margin,
                self.margin,
                self.margin,
                self.margin
            )
        )

        self.right_title = gui.Label(
            "Images contenant le point sélectionné"
        )
        self.right_panel_background.add_child(
            self.right_title
        )

        self.right_panel = gui.ScrollableVert(
            0.25 * self.em,
            gui.Margins(
                self.margin,
                self.margin,
                self.margin,
                self.margin
            )
        )

        self.right_items_layout = gui.VGrid(
            3,
            0.35 * self.em,
            gui.Margins(0, 0, 0, 0)
        )

        self.right_panel.add_child(
            self.right_items_layout
        )
        self.right_panel_background.add_child(
            self.right_panel
        )

        self.panel_width = int(22 * self.em)
        self.right_width = int(42 * self.em)

        self.state = {
            "point_size": float(point_size),
            "frustum_scale": float(frustum_scale),
            "show_pointcloud": True,
            "show_cameras": True,
            "show_camera_images": True,
            "only_colmap_observations": True,
            "show_only_tracked_points": False,
            "point_size_pending": float(point_size),
            "frustum_scale_pending": float(frustum_scale),
        }

        dbg(
            f"Point observations chargées: "
            f"{len(self.point_observations)} points trackés",
            self.verbose,
        )

        self._build_ui()
        self._populate_scene()
        self._install_interaction_handlers()
        self._refresh_image_list_for_selection()
        self._on_recenter()

    def _get_displayed_points_and_ids(self):
        if self.state.get("show_only_tracked_points", False):
            return self.tracked_points_xyz, self.tracked_point_ids
        return self.points_xyz, self.point_ids

    def _compute_camera_scale_defaults(self):
        centers = [build_T_wc_from_colmap_image(im)[:3, 3] for im in self.colmap_images.values()]
        if len(centers) < 2:
            print("[CAMSCALE] Trop peu de centres de caméras, fallback.")
            return 1.0, 1.0

        centers = np.asarray(centers, dtype=np.float64)
        scene_center = centers.mean(axis=0)
        dists = np.linalg.norm(centers - scene_center, axis=1)
        max_dist = np.max(dists)
        print(f"[CAMSCALE] scene_center={scene_center} max_dist={max_dist:.6f}")

        depth = 0.10 * max_dist
        angle_deg = 45.0
        half_angle_rad = np.deg2rad(angle_deg / 2.0)
        base_width = 2.0 * depth * np.tan(half_angle_rad)

        print(f"[CAMSCALE] FOND CHAMBRE : depth={depth:.6f}m")
        print(f"[CAMSCALE] angle sommet frustum={angle_deg:.2f}°, base_width={base_width:.6f}")

        return depth, base_width

    def _build_ui(self):
        self.left_panel.add_child(gui.Label("3D scene"))

        self.pointcloud_checkbox = gui.Checkbox("Display point cloud")
        self.pointcloud_checkbox.checked = True
        self.pointcloud_checkbox.set_on_checked(self._on_toggle_pointcloud)
        self.left_panel.add_child(self.pointcloud_checkbox)

        self.camera_checkbox = gui.Checkbox("Display cameras")
        self.camera_checkbox.checked = True
        self.camera_checkbox.set_on_checked(self._on_toggle_cameras)
        self.left_panel.add_child(self.camera_checkbox)

        self.camera_images_checkbox = gui.Checkbox("Display images")
        self.camera_images_checkbox.checked = True
        self.camera_images_checkbox.set_on_checked(self._on_toggle_camera_images)
        self.left_panel.add_child(self.camera_images_checkbox)

        self.colmap_obs_checkbox = gui.Checkbox("COLMAP observations only")
        self.colmap_obs_checkbox.checked = True
        self.colmap_obs_checkbox.set_on_checked(self._on_toggle_colmap_observations)
        self.left_panel.add_child(self.colmap_obs_checkbox)

        self.tracked_points_checkbox = gui.Checkbox("Display only tracked 3D points")
        self.tracked_points_checkbox.checked = False
        self.tracked_points_checkbox.set_on_checked(self._on_toggle_tracked_points)
        self.left_panel.add_child(self.tracked_points_checkbox)

        self.left_panel.add_child(gui.Label("3D points size"))
        self.pointsize_slider = gui.Slider(gui.Slider.DOUBLE)
        self.pointsize_slider.set_limits(1.0, 10.0)
        self.pointsize_slider.double_value = self.state["point_size"]
        self.pointsize_slider.set_on_value_changed(self._on_point_size_changed)
        self.left_panel.add_child(self.pointsize_slider)

        self.left_panel.add_child(gui.Label("Camera size"))
        self.frustum_slider = gui.Slider(gui.Slider.DOUBLE)
        self.frustum_slider.set_limits(self.camera_scale_min, self.camera_scale_max)
        self.frustum_slider.double_value = self.state["frustum_scale"]
        self.frustum_slider.set_on_value_changed(self._on_frustum_scale_changed)
        self.left_panel.add_child(self.frustum_slider)

        self.left_panel.add_child(gui.Label(
            f"Base={self.frustum_basewidth:.3f} | depth={self.frustum_depth:.3f} | factor=[{self.camera_scale_min:.2f}, {self.camera_scale_max:.2f}]"
        ))

        self.apply_button = gui.Button("Apply new scales")
        self.apply_button.set_on_clicked(self._on_apply_settings)
        self.left_panel.add_child(self.apply_button)

        self.recenter_button = gui.Button("Recenter view")
        self.recenter_button.set_on_clicked(self._on_recenter)
        self.left_panel.add_child(self.recenter_button)

        self.left_panel.add_child(gui.Label("Selection"))
        self.left_panel.add_child(gui.Label("Clic: nearest visible point (buffer Z)"))

        self.selection_label = gui.Label("No selected point")
        self.left_panel.add_child(self.selection_label)

        self.window.add_child(self.left_panel)
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.right_panel_background)

        def _on_layout(ctx):
            rect = self.window.content_rect
            self.left_panel.frame = gui.Rect(rect.x, rect.y, self.panel_width, rect.height)
            self.right_panel_background.frame = gui.Rect(
                rect.get_right() - self.right_width, rect.y, self.right_width, rect.height
            )
            center_x = rect.x + self.panel_width
            center_w = rect.width - self.panel_width - self.right_width
            self.scene_widget.frame = gui.Rect(center_x, rect.y, center_w, rect.height)

        self.window.set_on_layout(_on_layout)

    def _project_points_to_screen(self, points_xyz):
        points_xyz = np.asarray(points_xyz, dtype=np.float64)
        if points_xyz.size == 0:
            return {
                "screen_xy": np.zeros((0, 2), dtype=np.float64),
                "depth": np.zeros((0,), dtype=np.float64),
                "valid": np.zeros((0,), dtype=bool),
            }

        cam = self.scene_widget.scene.camera
        view = np.asarray(cam.get_view_matrix(), dtype=np.float64)
        proj = np.asarray(cam.get_projection_matrix(), dtype=np.float64)

        frame = self.scene_widget.frame
        width = max(int(frame.width), 1)
        height = max(int(frame.height), 1)

        pts_h = np.concatenate(
            [points_xyz, np.ones((len(points_xyz), 1), dtype=np.float64)],
            axis=1
        )

        clip = (proj @ view @ pts_h.T).T
        w = clip[:, 3]

        valid = np.abs(w) > 1e-12
        ndc = np.zeros((len(points_xyz), 3), dtype=np.float64)
        ndc[valid] = clip[valid, :3] / w[valid, None]

        inside = (
            valid &
            (clip[:, 3] > 0.0) &
            (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0) &
            (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0) &
            (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
        )

        sx = frame.x + (ndc[:, 0] * 0.5 + 0.5) * width
        sy = frame.y + (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height

        screen_xy = np.stack([sx, sy], axis=1)

        return {
            "screen_xy": screen_xy,
            "depth": ndc[:, 2],
            "valid": inside,
        }

    def _read_depth_at_click(self, x, y):
        depth_img = None
        try:
            depth_img = self.scene_widget.scene.scene.render_to_depth_image(
                z_in_view_space=True
            )
        except TypeError:
            try:
                depth_img = self.scene_widget.scene.scene.render_to_depth_image()
            except Exception as e:
                dbg(f"render_to_depth_image() indisponible: {e}", self.verbose)
                return None
        except Exception as e:
            dbg(f"Impossible de récupérer le buffer Z: {e}", self.verbose)
            return None

        if depth_img is None:
            return None

        depth = np.asarray(depth_img)
        if depth.ndim != 2 or depth.size == 0:
            return None

        frame = self.scene_widget.frame
        px = int(round(x - frame.x))
        py = int(round(y - frame.y))

        if px < 0 or py < 0 or px >= depth.shape[1] or py >= depth.shape[0]:
            dbg(f"Clic hors buffer depth: px={px}, py={py}, shape={depth.shape}", self.verbose)
            return None

        d = float(depth[py, px])

        if not np.isfinite(d):
            dbg("Depth non finie au clic", self.verbose)
            return None

        if d <= 0.0:
            dbg(f"Depth invalide au clic: {d}", self.verbose)
            return None

        dbg(f"Depth au clic: {d}", self.verbose)
        return d

    def _select_nearest_point_from_click(self, x, y):
        dbg(f"Clic dans la vue 3D: x={x}, y={y}", self.verbose)

        displayed_xyz, displayed_ids = self._get_displayed_points_and_ids()

        if len(displayed_xyz) == 0:
            warn("Aucun point 3D affiché.")
            return

        proj = self._project_points_to_screen(displayed_xyz)
        screen_xy = proj["screen_xy"]
        depth = proj["depth"]
        valid = proj["valid"]

        visible_count = int(np.count_nonzero(valid))
        dbg(f"Points visibles dans la vue courante: {visible_count}", self.verbose)

        if not np.any(valid):
            warn("Aucun point visible dans la vue courante.")
            return

        pts2d = screen_xy[valid]
        ptsz = depth[valid]
        valid_indices = np.flatnonzero(valid)

        target = np.array([[x, y]], dtype=np.float64)
        d2 = np.sum((pts2d - target) ** 2, axis=1)

        click_depth = self._read_depth_at_click(x, y)

        pixel_radius = 12.0
        local_mask = d2 <= (pixel_radius * pixel_radius)

        if np.any(local_mask):
            local_indices = np.flatnonzero(local_mask)

            if click_depth is not None:
                local_depth = ptsz[local_mask]
                depth_diff = np.abs(local_depth - click_depth)

                best_local_rel = int(np.argmin(depth_diff))
                best_local = local_indices[best_local_rel]

                dbg(
                    f"Sélection via buffer Z: idx={valid_indices[best_local]}, "
                    f"d2={d2[best_local]:.3f}, depth_pt={ptsz[best_local]:.6f}, depth_click={click_depth:.6f}",
                    self.verbose,
                )
            else:
                best_local_rel = int(np.argmin(d2[local_mask]))
                best_local = local_indices[best_local_rel]

                dbg(
                    f"Fallback 2D local: idx={valid_indices[best_local]}, "
                    f"d2={d2[best_local]:.3f}, depth_pt={ptsz[best_local]:.6f}",
                    self.verbose,
                )

            idx_displayed = int(valid_indices[best_local])
        else:
            idx_local = int(np.argmin(d2))
            idx_displayed = int(valid_indices[idx_local])

            dbg(
                f"Fallback 2D global: idx={idx_displayed}, "
                f"d2={d2[idx_local]:.3f}, depth_pt={ptsz[idx_local]:.6f}",
                self.verbose,
            )

        selected_pid = int(displayed_ids[idx_displayed])

        idx_global = self.point_id_to_global_index.get(selected_pid)
        if idx_global is None:
            warn(f"Impossible de retrouver l'index global pour point_id={selected_pid}")
            return

        self._set_selected_point(idx_global)

    def _install_interaction_handlers(self):
        def _on_mouse(event):
            if event.type == gui.MouseEvent.BUTTON_UP and event.is_modifier_down(gui.KeyModifier.SHIFT):
                self._select_nearest_point_from_click(event.x, event.y)
                return gui.Widget.EventCallbackResult.HANDLED

            return gui.Widget.EventCallbackResult.IGNORED

        self.scene_widget.set_on_mouse(_on_mouse)

    def _build_colored_pointcloud(self):
        points_xyz, _ = self._get_displayed_points_and_ids()

        pcd = o3d.geometry.PointCloud()
        if len(points_xyz) == 0:
            return pcd

        pcd.points = o3d.utility.Vector3dVector(points_xyz)

        if self.state.get("show_only_tracked_points", False):
            if self.tracked_point_colors is not None and len(self.tracked_point_colors) == len(points_xyz):
                pcd.colors = o3d.utility.Vector3dVector(self.tracked_point_colors)
        else:
            if self.raw_point_colors is not None and len(self.raw_point_colors) == len(points_xyz):
                pcd.colors = o3d.utility.Vector3dVector(self.raw_point_colors)

        return colorize_point_cloud_by_z(pcd)

    def _populate_scene(self):
        try:
            self.scene_widget.scene.remove_geometry("pointcloud")
        except Exception:
            pass

        for name in list(getattr(self, "_camera_geom_names", [])):
            try:
                self.scene_widget.scene.remove_geometry(name)
            except Exception:
                pass
        self._camera_geom_names = []

        for name in list(getattr(self, "_camera_image_geom_names", [])):
            try:
                self.scene_widget.scene.remove_geometry(name)
            except Exception:
                pass
        self._camera_image_geom_names = []

        try:
            self.scene_widget.scene.remove_geometry(self.selected_sphere_name)
        except Exception:
            pass

        if self.state["show_pointcloud"]:
            pcd = self._build_colored_pointcloud()
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.point_size = self.state["point_size"]
            self.scene_widget.scene.add_geometry("pointcloud", pcd, mat)

        if self.state["show_cameras"]:
            current_base_width = self.frustum_basewidth * self.state["frustum_scale"]
            current_depth = self.frustum_depth * self.state["frustum_scale"]

            for image_id, im in sorted(self.colmap_images.items()):
                T_wc = build_T_wc_from_colmap_image(im)
                cam = self.colmap_cameras.get(im["camera_id"])
                aspect = 1.0
                if cam is not None and cam["height"] > 0:
                    aspect = cam["width"] / cam["height"]

                frustum = create_camera_frustum(
                    T_wc,
                    depth=current_depth,
                    base_width=current_base_width,
                    aspect=aspect,
                    color=(1.0, 0.0, 0.0),
                )

                name = f"cam_{image_id}"
                mat_line = rendering.MaterialRecord()
                mat_line.shader = "unlitLine"
                mat_line.line_width = 2.0
                self.scene_widget.scene.add_geometry(name, frustum, mat_line)
                self._camera_geom_names.append(name)

                image_path = resolve_image_path(
                    self.images_dir,
                    im["name"],
                    preview_cache_dir=self.preview_cache_dir,
                    make_preview_script=self.make_preview_script,
                    preview_max_size=self.preview_max_size,
                    verbose=self.verbose,
                )

                if self.state["show_camera_images"]:
                    if image_path is not None:
                        try:
                            with Image.open(image_path) as img:
                                w, h = img.size
                            if h > 0 and w > 0:
                                aspect = w / h

                            quad = create_textured_image_quad(
                                T_wc,
                                depth=current_depth,
                                base_width=current_base_width,
                                aspect=aspect,
                            )

                            material = rendering.MaterialRecord()
                            material.shader = "defaultUnlit"
                            material.base_color = [1.0, 1.0, 1.0, 1.0]
                            material.albedo_img = o3d.io.read_image(str(image_path))

                            img_name = f"cam_img_{image_id}"
                            self.scene_widget.scene.add_geometry(img_name, quad, material)
                            self._camera_image_geom_names.append(img_name)
                        except Exception as e:
                            dbg(f"Impossible d'ajouter la preview caméra {im['name']}: {e}", self.verbose)

        self._update_selection_geometry()

    def _compute_scene_bounds(self):
        arrays = []

        displayed_xyz, _ = self._get_displayed_points_and_ids()
        if len(displayed_xyz) > 0:
            arrays.append(displayed_xyz)

        centers = []
        for im in self.colmap_images.values():
            T_wc = build_T_wc_from_colmap_image(im)
            centers.append(T_wc[:3, 3])
        if centers:
            arrays.append(np.asarray(centers, dtype=np.float64))

        if self.selected_point_xyz is not None:
            arrays.append(self.selected_point_xyz.reshape(1, 3))

        if not arrays:
            return None

        pts = np.vstack(arrays)
        pmin = pts.min(axis=0)
        pmax = pts.max(axis=0)
        pad = np.maximum((pmax - pmin) * 0.05, 1e-3)

        return o3d.geometry.AxisAlignedBoundingBox(pmin - pad, pmax + pad)

    def _estimate_camera_scale(self):
        if len(self.colmap_images) < 2:
            return 1.0

        centers = []
        for im in self.colmap_images.values():
            T_wc = build_T_wc_from_colmap_image(im)
            centers.append(T_wc[:3, 3])

        centers = np.asarray(centers, dtype=np.float64)

        cmin = centers.min(axis=0)
        cmax = centers.max(axis=0)

        bbox_size = cmax - cmin
        extent = np.linalg.norm(bbox_size)

        if extent < 1e-9:
            return 1.0

        scale = extent * 0.5

        return float(scale)

    def _set_selected_point(self, idx):
        idx = int(idx)
        if idx < 0 or idx >= len(self.points_xyz):
            warn(f"Index point invalide: {idx}")
            return

        self.selected_point_index = idx
        self.selected_point_xyz = self.points_xyz[idx].copy()

        pid = int(self.point_ids[idx]) if idx < len(self.point_ids) else idx
        p = self.selected_point_xyz
        self.selection_label.text = (
            f"1 point sélectionné: index={idx}, id={pid}, "
            f"XYZ=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"
        )
        info(self.selection_label.text)

        self._update_selection_geometry()
        self._refresh_image_list_for_selection()

    def _update_selection_geometry(self):
        try:
            self.scene_widget.scene.remove_geometry(self.selected_sphere_name)
        except Exception:
            pass

        if self.selected_point_xyz is None:
            return

        scene_extent = 1.0
        displayed_xyz, _ = self._get_displayed_points_and_ids()
        if len(displayed_xyz) > 0:
            pmin = displayed_xyz.min(axis=0)
            pmax = displayed_xyz.max(axis=0)
            scene_extent = max(float(np.max(pmax - pmin)), 1e-6)

        radius = max(scene_extent * 0.0001, 1e-5)
        sphere = make_sphere(self.selected_point_xyz, radius=radius, color=(1.0, 1.0, 0.0))

        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        self.scene_widget.scene.add_geometry(self.selected_sphere_name, sphere, mat)

    def _on_toggle_pointcloud(self, checked):
        self.state["show_pointcloud"] = bool(checked)
        self._populate_scene()

    def _on_toggle_cameras(self, checked):
        self.state["show_cameras"] = bool(checked)
        self._populate_scene()

    def _on_toggle_camera_images(self, checked):
        self.state["show_camera_images"] = bool(checked)
        self._populate_scene()

    def _on_toggle_colmap_observations(self, checked):
        self.state["only_colmap_observations"] = bool(checked)
        self._refresh_image_list_for_selection()

    def _on_toggle_tracked_points(self, checked):
        self.state["show_only_tracked_points"] = bool(checked)
        self._populate_scene()

    def _on_point_size_changed(self, value):
        self.state["point_size_pending"] = float(value)

    def _on_frustum_scale_changed(self, value):
        self.state["frustum_scale_pending"] = float(value)

    def _on_apply_settings(self):
        self.state["point_size"] = self.state["point_size_pending"]
        self.state["frustum_scale"] = self.state["frustum_scale_pending"]

        self._populate_scene()

    def _on_recenter(self):
        centers = []

        for colmap_image in self.colmap_images.values():
            T_wc = build_T_wc_from_colmap_image(colmap_image)
            c = T_wc[:3, 3].copy()
            centers.append(c)

        if not centers:
            warn("Impossible de recentrer: aucune position caméra disponible.")
            return

        centers = np.asarray(centers, dtype=np.float64)
        cmin = centers.min(axis=0)
        cmax = centers.max(axis=0)
        bbox_center = 0.5 * (cmin + cmax)
        bbox_extent = cmax - cmin
        bbox_radius = 0.5 * np.linalg.norm(bbox_extent)

        if bbox_radius < 1e-6:
            bbox_radius = 1.0

        try:
            cam = self.scene_widget.scene.camera

            model = np.asarray(cam.get_model_matrix(), dtype=np.float64)
            if model.shape != (4, 4):
                raise ValueError(
                    f"get_model_matrix() retourne une matrice de forme inattendue: {model.shape}"
                )

            cam_right = model[:3, 0]
            cam_up = model[:3, 1]
            cam_forward = model[:3, 2]
            cam_pos = model[:3, 3]

            def _safe_normalize(v, fallback):
                n = np.linalg.norm(v)
                if n < 1e-12:
                    return np.asarray(fallback, dtype=np.float64)
                return v / n

            cam_right = _safe_normalize(cam_right, [1.0, 0.0, 0.0])
            cam_up = _safe_normalize(cam_up, [0.0, 1.0, 0.0])
            cam_forward = _safe_normalize(cam_forward, [0.0, 0.0, 1.0])

            view_dir = -cam_forward

            delta = bbox_center - cam_pos

            depth = np.dot(delta, view_dir)
            x_offset = np.dot(delta, cam_right)
            y_offset = np.dot(delta, cam_up)

            lateral_shift = x_offset * cam_right + y_offset * cam_up

            fov_deg = float(cam.get_field_of_view())
            fov_rad = np.deg2rad(fov_deg)
            target_dist = bbox_radius / max(np.tan(fov_rad * 0.5), 1e-6) * 1.2
            target_dist = max(target_dist, 1e-3)

            new_eye = cam_pos + lateral_shift + (depth - target_dist) * view_dir
            orbit_center = bbox_center

            self.scene_widget.look_at(orbit_center, new_eye, cam_up)

            try:
                near = max(target_dist * 0.01, 0.01)
                far = max(target_dist + 20.0 * bbox_radius, 50.0)

                self.scene_widget.scene.camera.set_projection(
                    fov_deg,
                    self.scene_widget.frame.width / max(self.scene_widget.frame.height, 1),
                    near,
                    far,
                    rendering.Camera.FovType.Vertical
                )
            except Exception:
                pass

            self.window.post_redraw()

        except Exception as e:
            warn(f"Impossible de recentrer la vue: {e}")

    def _clear_right_panel(self):
        if hasattr(self.right_items_layout, "clear_children"):
            self.right_items_layout.clear_children()
            return

        try:
            self.right_items_layout.visible = False
        except Exception:
            pass

        self.right_items_layout = gui.VGrid(
            3,
            0.35 * self.em,
            gui.Margins(0, 0, 0, 0)
        )
        self.right_panel.add_child(self.right_items_layout)

    def _make_image_card(self, item):
        block = gui.Vert(
            0.15 * self.em,
            gui.Margins(0.2 * self.em, 0.2 * self.em, 0.2 * self.em, 0.2 * self.em)
        )

        title = gui.Label(f"{item['name']}")
        block.add_child(title)

        subtitle = gui.Label(f"depth={item['depth']:.2f}")
        block.add_child(subtitle)

        try:
            dbg(f"Ouverture image pour affichage: {item['image_path']}", self.verbose)
            annotated = draw_zoomed_projections_on_image(
                item["image_path"],
                item["uv"],
                item["width"],
                item["height"],
                target_size=200,
                padding_px=30,
                marker_radius=5,
            )
            dbg(f"Image annotée OK: {item['name']}", self.verbose)
            o3d_img = pil_to_o3d_image(annotated)
            gui_img = gui.ImageWidget(o3d_img)
            block.add_child(gui_img)
        except Exception as e:
            warn(f"Impossible d'annoter ou afficher l'image {item['name']}: {e}")
            block.add_child(gui.Label(f"Erreur image: {e}"))

        return block

    def _refresh_image_list_for_selection(self):
        self._clear_right_panel()

        if self.selected_point_xyz is None:
            info("Aucun point sélectionné: aucune image à afficher.")
            self.right_items_layout.add_child(
                gui.Label("Aucun point sélectionné.")
            )
            return

        if self.selected_point_index is None:
            warn("selected_point_index=None")
            return

        point_id = int(self.point_ids[self.selected_point_index])

        if self.state["only_colmap_observations"]:
            obs = self.point_observations.get(point_id)
            candidate_image_ids = (
                obs.get("image_ids", [])
                if obs is not None
                else []
            )

            info(
                f"Mode observations COLMAP : "
                f"point_id={point_id} "
                f"({len(candidate_image_ids)} image(s))"
            )
        else:
            candidate_image_ids = list(self.colmap_images.keys())

            info(
                f"Mode toutes les images : "
                f"point_id={point_id} "
                f"({len(candidate_image_ids)} image(s) testées)"
            )

        projections_per_image = []

        if not candidate_image_ids:
            warn(
                f"Aucune observation COLMAP "
                f"pour point_id={point_id}"
            )
            self.right_items_layout.add_child(
                gui.Label(
                    "Aucune image ne contient ce point."
                )
            )
            return

        for image_id in candidate_image_ids:
            colmap_image = self.colmap_images.get(image_id)

            if colmap_image is None:
                warn(
                    f"Image COLMAP absente: "
                    f"image_id={image_id}"
                )
                continue

            cam = self.colmap_cameras.get(colmap_image["camera_id"])

            if cam is None:
                warn(
                    f"Caméra introuvable "
                    f"pour image_id={image_id}"
                )
                continue

            image_name = colmap_image["name"]

            dbg(
                f"Test image trackée: "
                f"id={image_id}, "
                f"name={image_name}",
                self.verbose,
            )

            image_path = resolve_image_path(
                self.images_dir,
                image_name,
                verbose=self.verbose,
            )

            if image_path is None:
                warn(
                    f"Image non trouvée "
                    f"dans images/: {image_name}"
                )
                continue

            dbg(
                f"Image résolue: {image_path}",
                self.verbose,
            )

            try:
                proj = project_world_point(
                    colmap_image,
                    cam,
                    self.selected_point_xyz,
                    z_positive=self.z_positive,
                )

            except NotImplementedError as e:
                warn(
                    f"Projection non supportée "
                    f"pour {image_name}: {e}"
                )
                proj = None

            if proj is None:
                dbg("projection=None", self.verbose)
                continue

            if not proj["inside"]:
                dbg(
                    f"Hors image "
                    f"u={proj['uv'][0]:.2f}, "
                    f"v={proj['uv'][1]:.2f}",
                    self.verbose,
                )
                continue

            dbg(
                f"Dedans "
                f"u={proj['uv'][0]:.2f}, "
                f"v={proj['uv'][1]:.2f}, "
                f"z={proj['depth']:.3f}",
                self.verbose,
            )

            projections_per_image.append({
                "image_id": image_id,
                "name": image_name,
                "image_path": image_path,
                "uv": proj["uv"].reshape(1, 2),
                "depth": float(proj["depth"]),
                "width": int(cam["width"]),
                "height": int(cam["height"]),
            })

            info(f"Image retenue: {image_name}")

        projections_per_image.sort(key=lambda x: x["depth"])

        info(
            f"Nombre total d'images "
            f"affichables: "
            f"{len(projections_per_image)}"
        )

        if not projections_per_image:
            warn(
                "Aucune image exploitable "
                "pour le point sélectionné"
            )

            self.right_items_layout.add_child(
                gui.Label(
                    "Aucune image ne contient "
                    "le point sélectionné."
                )
            )
            return

        header = gui.Label(
            f"{len(projections_per_image)} "
            f"image(s) correspondante(s)"
        )

        self.right_items_layout.add_child(header)
        self.right_items_layout.add_child(gui.Label(""))
        self.right_items_layout.add_child(gui.Label(""))

        cards = [
            self._make_image_card(item)
            for item in projections_per_image
        ]

        remainder = len(cards) % 3

        if remainder != 0:
            for _ in range(3 - remainder):
                cards.append(gui.Label(""))

        for card in cards:
            self.right_items_layout.add_child(card)

    def run(self):
        self.app.run()


def main():
    ap = argparse.ArgumentParser(description="Interface 3D/2D d'inspection photogrammétrique")
    ap.add_argument("--colmap-dir", required=True, help="Répertoire racine contenant colmap/, images/")
    ap.add_argument("--point-size", type=float, default=2.0, help="Taille d'affichage des points")
    ap.add_argument(
        "--frustum-scale",
        type=float,
        default=None,
        help="Taille relative des caméras. Si omis, valeur auto basée sur l'espacement moyen entre poses.",
    )
    ap.add_argument(
        "--z-negative",
        action="store_true",
        help="Utilise la convention profondeur z<0 (par défaut: z>0)."
    )
    ap.add_argument("--verbose", action="store_true", help="Logs détaillés")
    args = ap.parse_args()

    colmap_dir = Path(args.colmap_dir)

    images_dir = colmap_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Dossier images introuvable: {images_dir}")

    info(f"Dossier images utilisé: {images_dir}")

    pointcloud_data = load_3dpoints(colmap_dir)
    info(
        f"Observations chargées: "
        f"{len(pointcloud_data.get('observations', {}))}"
    )
    colmap_images = load_colmap_images(colmap_dir)
    colmap_cameras = load_colmap_cameras(colmap_dir)

    z_positive = not args.z_negative
    info(f"Convention profondeur: {'z>0' if z_positive else 'z<0'}")

    app = InspectorApp(
        pointcloud_data=pointcloud_data,
        colmap_images=colmap_images,
        colmap_cameras=colmap_cameras,
        colmap_dir=colmap_dir,
        point_size=args.point_size,
        frustum_scale=args.frustum_scale,
        z_positive=z_positive,
        verbose=args.verbose,
    )
    app.run()


if __name__ == "__main__":
    main()

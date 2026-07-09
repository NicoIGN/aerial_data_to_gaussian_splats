#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
import sys
from datetime import datetime
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import subprocess
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import laspy
except ImportError:
    print("Erreur: laspy requis. Installe: pip install laspy[lazrs]")
    sys.exit(1)

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    print("Erreur: scipy requis. Installe: pip install scipy")
    sys.exit(1)


IMAGE_EXTS = {".jp2", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Occlusion mode: paramètres internes (volontairement non exposés en CLI)
OCCLUSION_RADIUS_PX = 1          # rayon splat autour du pixel projeté
OCCLUSION_TOL_ABS = 0.10         # tolérance absolue profondeur (en unités scène)
OCCLUSION_TOL_REL = 0.01         # tolérance relative profondeur


def log(msg: str, level: int, verbose: int):
    if verbose >= level:
        print(msg)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def build_image_index(image_dir: Path):
    by_stem = {}
    by_name = {}

    for p in sorted(image_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue

        by_name.setdefault(p.name, p)
        by_stem.setdefault(p.stem, p)

    return by_stem, by_name


def resolve_con_path_for_image(img_path: Path, image_dir: Path):
    c1 = img_path.with_suffix(".CON")
    if c1.exists():
        return c1

    c2 = img_path.with_suffix(".con")
    if c2.exists():
        return c2

    stem = img_path.stem
    candidates = []
    for p in image_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".con"}:
            continue
        if p.stem == stem:
            candidates.append(p)

    if candidates:
        return sorted(candidates)[0]

    return None


def read_laz_points(laz_path: Path, stride: int):
    las = laspy.read(str(laz_path))
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)

    rgb = None
    has_rgb = all(hasattr(las, a) for a in ("red", "green", "blue"))
    if has_rgb:
        rgb = np.vstack((las.red, las.green, las.blue)).T
        if rgb.size > 0:
            if rgb.max() > 255:
                rgb = np.clip(rgb / 256.0, 0, 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)

    stride = max(1, int(stride))
    xyz = xyz[::stride]
    if rgb is not None:
        rgb = rgb[::stride]

    return xyz, rgb


def filter_xy_bbox(xyz: np.ndarray, rgb: np.ndarray | None, xmin=None, xmax=None, ymin=None, ymax=None):
    if xmin is None and xmax is None and ymin is None and ymax is None:
        return xyz, rgb

    mask = np.ones(len(xyz), dtype=bool)

    if xmin is not None:
        mask &= xyz[:, 0] >= float(xmin)
    if xmax is not None:
        mask &= xyz[:, 0] <= float(xmax)
    if ymin is not None:
        mask &= xyz[:, 1] >= float(ymin)
    if ymax is not None:
        mask &= xyz[:, 1] <= float(ymax)

    xyz2 = xyz[mask]
    rgb2 = None if rgb is None else rgb[mask]
    return xyz2, rgb2


def get_nerfstudio_axis_transform_4x4():
    T = np.eye(4, dtype=np.float64)
    T[:3, :4] = np.array([
        [1.0,  0.0,  0.0, 0.0],
        [0.0,  0.0,  1.0, 0.0],
        [0.0, -1.0,  0.0, 0.0],
    ], dtype=np.float64)
    return T


def write_ply_xyzrgb(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None, verbose: int = 1):
    n = len(xyz)
    if rgb is None:
        rgb = np.full((n, 3), 200, dtype=np.uint8)
    else:
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    xyz_out = np.asarray(xyz, dtype=np.float64)

    use_tqdm = verbose >= 1 and "tqdm" in globals() and tqdm is not None
    iterable = zip(xyz_out, rgb)

    if use_tqdm:
        iterable = tqdm(
            iterable,
            total=n,
            desc="Écriture sparse_pc.ply",
            unit="pt",
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in iterable:
            f.write(
                f"{float(p[0]):.12f} {float(p[1]):.12f} {float(p[2]):.12f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def prepare_selected_images(source_paths, output_dir: Path, factor: float, jpeg_quality: int,
                            verbose: int, image_convertor_script: Path):
    ensure_dir(output_dir)

    if not image_convertor_script.exists():
        raise FileNotFoundError(f"Script de conversion introuvable: {image_convertor_script}")

    produced = {}

    for src_path in source_paths:
        src_path = Path(src_path)
        stem = src_path.stem
        ext = src_path.suffix.lower()

        if ext in {".jp2", ".tif", ".tiff"}:
            out_path = output_dir / f"{stem}.jpg"
            output_mode = "--jpg"

        elif ext in {".jpg", ".jpeg", ".png", ".bmp"}:
            if factor <= 1.0:
                out_path = output_dir / src_path.name
                log(f"  Copie: {src_path.name}", 2, verbose)
                shutil.copy2(src_path, out_path)
                produced[stem] = out_path
                continue

            out_path = output_dir / src_path.name
            if ext in {".jpg", ".jpeg"}:
                output_mode = "--jpg"
            elif ext == ".png":
                output_mode = "--png"
            elif ext == ".bmp":
                output_mode = "--png"
                out_path = output_dir / f"{stem}.png"
            else:
                output_mode = "--jpg"
                out_path = output_dir / f"{stem}.jpg"
        else:
            log(f"[WARN] Format non géré ignoré: {src_path.name}", 2, verbose)
            continue

        cmd = [
            sys.executable,
            str(image_convertor_script),
            "--input", str(src_path),
            "--output", str(out_path),
            "--factor", str(factor),
            output_mode,
            "--jpeg-quality", str(jpeg_quality),
        ]

        if verbose >= 1:
            cmd.append("--verbose")

        log(f"  Conversion: {src_path.name} -> {out_path.name}", 2, verbose)
        subprocess.run(cmd, check=True)

        if not out_path.exists():
            raise FileNotFoundError(f"Image convertie introuvable après conversion: {out_path}")

        produced[stem] = out_path

    return produced


def apply_cylindrical_systematism_local_to_image(c, l, transfo2d):
    if transfo2d is None:
        return c, l

    tr_type = transfo2d.get("Type")
    if tr_type != "systematismeCylindriqueTopAero":
        return c, l

    C0 = float(transfo2d.get("C0", 0.0))
    L0 = float(transfo2d.get("L0", 0.0))
    S1 = float(transfo2d.get("S1", 0.0))
    S2 = float(transfo2d.get("S2", 0.0))

    ci = c
    li = l

    l = li + (ci - C0) * S1
    c = ci + (ci - C0) * S2

    return c, l


def apply_cylindrical_systematism_image_to_local(c, l, transfo2d):
    if transfo2d is None:
        return c, l

    tr_type = transfo2d.get("Type")
    if tr_type != "systematismeCylindriqueTopAero":
        return c, l

    C0 = float(transfo2d.get("C0", 0.0))
    L0 = float(transfo2d.get("L0", 0.0))
    S1 = float(transfo2d.get("S1", 0.0))
    S2 = float(transfo2d.get("S2", 0.0))

    ci = c
    li = l

    l = li - (ci - C0) * S1
    c = ci - (ci - C0) * S2

    return c, l


def parse_con_orientation(con_path: Path):
    tree = ET.parse(con_path)
    root = tree.getroot()

    geometry = root.find("geometry")
    if geometry is None:
        raise ValueError(f"{con_path}: balise <geometry> absente")

    extr = geometry.find("extrinseque")
    intr = geometry.find("intrinseque")
    if extr is None or intr is None:
        raise ValueError(f"{con_path}: balises extrinseque/intrinseque absentes")

    sensor = intr.find("sensor")
    if sensor is None:
        raise ValueError(f"{con_path}: balise intrinseque/sensor absente")

    eucl = extr.find("systeme/euclidien")
    sommet = extr.find("sommet")
    rotation = extr.find("rotation")

    if eucl is None or sommet is None or rotation is None:
        raise ValueError(f"{con_path}: extrinsèque incomplet")

    eucl_x = float(eucl.findtext("x"))
    eucl_y = float(eucl.findtext("y"))

    easting = float(sommet.findtext("easting"))
    northing = float(sommet.findtext("northing"))
    altitude = float(sommet.findtext("altitude"))

    center = np.array([
        eucl_x + easting,
        eucl_y + northing,
        altitude,
    ], dtype=np.float64)

    image2ground_txt = rotation.findtext("Image2Ground")
    image2ground = str(image2ground_txt).strip().lower() == "true"

    def _read_row(row_tag):
        row = rotation.find(f"mat3d/{row_tag}/pt3d")
        if row is None:
            raise ValueError(f"{con_path}: ligne {row_tag} absente dans rotation/mat3d")
        return [
            float(row.findtext("x")),
            float(row.findtext("y")),
            float(row.findtext("z")),
        ]

    M = np.array([
        _read_row("l1"),
        _read_row("l2"),
        _read_row("l3"),
    ], dtype=np.float64)

    if image2ground:
        R_cw = M.T
    else:
        R_cw = M

    det = np.linalg.det(R_cw)
    if det <= 0:
        raise ValueError(f"{con_path}: rotation invalide, det={det}")

    tvec = -R_cw @ center

    rot_final = R.from_matrix(R_cw)
    qx, qy, qz, qw = rot_final.as_quat()
    qvec = np.array([qw, qx, qy, qz], dtype=np.float64)

    width = int(sensor.findtext("image_size/width"))
    height = int(sensor.findtext("image_size/height"))
    cx = float(sensor.findtext("ppa/c"))
    cy = float(sensor.findtext("ppa/l"))
    fx = float(sensor.findtext("ppa/focale"))
    fy = fx

    pixel_size_txt = sensor.findtext("pixel_size")
    pixel_size = float(pixel_size_txt) if pixel_size_txt is not None else None

    transfo2d = None
    tr = sensor.find("transfo2d/tr2delem")
    if tr is not None:
        transfo2d = {
            "Type": tr.attrib.get("Type"),
            "isinterne": tr.attrib.get("isinterne"),
            "C0": float(tr.attrib.get("C0", "0")),
            "L0": float(tr.attrib.get("L0", "0")),
            "S1": float(tr.attrib.get("S1", "0")),
            "S2": float(tr.attrib.get("S2", "0")),
        }

    return {
        "center": center,
        "R_cw": R_cw,
        "tvec": tvec,
        "qvec": qvec,
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "pixel_size": pixel_size,
        "image2ground": image2ground,
        "transfo2d": transfo2d,
    }


def scale_intrinsics(width, height, fx, fy, cx, cy, factor):
    factor = max(float(factor), 1.0)
    width2 = max(1, int(round(width / factor)))
    height2 = max(1, int(round(height / factor)))
    fx2 = fx / factor
    fy2 = fy / factor
    cx2 = cx / factor
    cy2 = cy / factor
    return width2, height2, fx2, fy2, cx2, cy2


def scale_transfo2d(transfo2d, factor):
    if transfo2d is None:
        return None

    factor = max(float(factor), 1.0)

    scaled = dict(transfo2d)
    scaled["C0"] = float(transfo2d.get("C0", 0.0)) / factor
    scaled["L0"] = float(transfo2d.get("L0", 0.0)) / factor
    scaled["S1"] = float(transfo2d.get("S1", 0.0))
    scaled["S2"] = float(transfo2d.get("S2", 0.0))

    return scaled


def project_point(R_cw, t_cw, fx, fy, cx, cy, xyz, transfo2d=None, z_positive=True):
    Xc = R_cw @ xyz + t_cw
    z_raw = float(Xc[2])

    if z_positive:
        if z_raw <= 1e-9:
            return None
        z = z_raw
    else:
        if z_raw >= -1e-9:
            return None
        z = -z_raw

    u = fx * (Xc[0] / z) + cx
    v = fy * (Xc[1] / z) + cy

    u, v = apply_cylindrical_systematism_local_to_image(u, v, transfo2d)
    return np.array([u, v], dtype=np.float64), z


def _camera_project_batch(fr, pts_xyz, z_positive=True):
    R_cw = fr["R_cw"]
    t_cw = fr["tvec"]
    fx, fy, cx, cy = fr["fx"], fr["fy"], fr["cx"], fr["cy"]
    width, height = fr["width"], fr["height"]
    transfo2d = fr.get("transfo2d")

    Xc = (pts_xyz @ R_cw.T) + t_cw.reshape(1, 3)
    z_raw = Xc[:, 2]

    if z_positive:
        mask_front = z_raw > 1e-9
        z = z_raw
    else:
        mask_front = z_raw < -1e-9
        z = -z_raw

    valid_idx = np.where(mask_front)[0]
    if len(valid_idx) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    Xc_v = Xc[valid_idx]
    z_v = z[valid_idx]

    u = fx * (Xc_v[:, 0] / z_v) + cx
    v = fy * (Xc_v[:, 1] / z_v) + cy

    if transfo2d is not None and transfo2d.get("Type") == "systematismeCylindriqueTopAero":
        C0 = float(transfo2d.get("C0", 0.0))
        S1 = float(transfo2d.get("S1", 0.0))
        S2 = float(transfo2d.get("S2", 0.0))
        ci = u.copy()
        li = v.copy()
        v = li + (ci - C0) * S1
        u = ci + (ci - C0) * S2

    in_img = (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
    keep = np.where(in_img)[0]
    if len(keep) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    valid_idx = valid_idx[keep]
    u = u[keep]
    v = v[keep]
    z_v = z_v[keep]
    return valid_idx, u, v, z_v


def _build_occlusion_depth_map_for_frame(fr, pts_xyz, z_positive, radius_px=1):
    h, w = fr["height"], fr["width"]
    depth = np.full((h, w), np.inf, dtype=np.float32)

    idx, u, v, z = _camera_project_batch(fr, pts_xyz, z_positive=z_positive)
    if len(idx) == 0:
        return depth

    x = np.floor(u).astype(np.int32)
    y = np.floor(v).astype(np.int32)
    z = z.astype(np.float32)

    for dy in range(-radius_px, radius_px + 1):
        yy = y + dy
        my = (yy >= 0) & (yy < h)
        if not np.any(my):
            continue

        yy2 = yy[my]
        x2 = x[my]
        z2 = z[my]

        for dx in range(-radius_px, radius_px + 1):
            xx = x2 + dx
            mx = (xx >= 0) & (xx < w)
            if not np.any(mx):
                continue

            np.minimum.at(depth, (yy2[mx], xx[mx]), z2[mx])

    return depth


def build_occlusion_depth_maps(frames, pts_xyz, z_positive=True, verbose=1):
    maps = {}
    use_tqdm = verbose >= 1 and "tqdm" in globals() and tqdm is not None
    iterable = frames
    if use_tqdm:
        iterable = tqdm(frames, total=len(frames), desc="Occlusion: build depth maps", unit="img")

    for fr in iterable:
        maps[fr["image_id"]] = _build_occlusion_depth_map_for_frame(
            fr,
            pts_xyz,
            z_positive=z_positive,
            radius_px=OCCLUSION_RADIUS_PX
        )
    return maps


def is_visible_with_occlusion(depth_map, u, v, z):
    h, w = depth_map.shape
    x = int(np.floor(u))
    y = int(np.floor(v))
    if x < 0 or x >= w or y < 0 or y >= h:
        return False

    zmin = float(depth_map[y, x])
    if not np.isfinite(zmin):
        return False

    tol = OCCLUSION_TOL_ABS + OCCLUSION_TOL_REL * zmin
    return z <= (zmin + tol)


def build_synthetic_observations(frames, pts_xyz, pts_rgb=None, num_terrain_points=None, verbose=1, z_positive=True, with_occlusion=False):
    num_pts_total = len(pts_xyz)

    if num_pts_total == 0:
        return {fr["image_id"]: [] for fr in frames}, []

    if num_terrain_points is None or num_terrain_points <= 0 or num_terrain_points >= num_pts_total:
        selected_indices = np.arange(num_pts_total, dtype=np.int64)
    else:
        selected_indices = np.linspace(
            0,
            num_pts_total - 1,
            num=num_terrain_points,
            dtype=np.int64
        )

    selected_indices = np.unique(selected_indices)

    observations_by_image = {fr["image_id"]: [] for fr in frames}
    tracks_by_point = [[] for _ in range(num_pts_total)]

    selected_count = len(selected_indices)
    pct = 100.0 * selected_count / num_pts_total if num_pts_total > 0 else 0.0

    log(
        f"  Reprojection de {selected_count} point(s) terrain sur {num_pts_total} disponible(s) "
        f"({pct:.2f}%) vers {len(frames)} image(s)...",
        1,
        verbose,
    )

    occlusion_maps = None
    if with_occlusion:
        log(
            f"  Occlusion activée (radius={OCCLUSION_RADIUS_PX}px, tol_abs={OCCLUSION_TOL_ABS}, tol_rel={OCCLUSION_TOL_REL})",
            1,
            verbose,
        )
        pts_for_depth = pts_xyz[selected_indices]
        occlusion_maps = build_occlusion_depth_maps(
            frames=frames,
            pts_xyz=pts_for_depth,
            z_positive=z_positive,
            verbose=verbose,
        )

    iterable = selected_indices
    use_tqdm = verbose >= 1 and "tqdm" in globals() and tqdm is not None
    if use_tqdm:
        iterable = tqdm(
            selected_indices,
            total=len(selected_indices),
            desc="Reprojection points terrain",
            unit="pt",
        )

    for local_count, pt_idx in enumerate(iterable, start=1):
        xyz = pts_xyz[pt_idx]
        point3d_id = int(pt_idx + 1)

        for fr in frames:
            image_id = fr["image_id"]
            R_cw = fr["R_cw"]
            t_cw = fr["tvec"]
            width = fr["width"]
            height = fr["height"]
            fx = fr["fx"]
            fy = fr["fy"]
            cx = fr["cx"]
            cy = fr["cy"]
            transfo2d = fr.get("transfo2d")

            proj = project_point(
                R_cw, t_cw, fx, fy, cx, cy, xyz,
                transfo2d=transfo2d,
                z_positive=z_positive,
            )

            if proj is None:
                continue

            uv, depth = proj
            u, v = uv

            if not (0.0 <= u < width and 0.0 <= v < height):
                continue

            if with_occlusion:
                depth_map = occlusion_maps[image_id]
                if not is_visible_with_occlusion(depth_map, u, v, depth):
                    continue

            point2d_idx = len(observations_by_image[image_id])

            observations_by_image[image_id].append({
                "xy": np.asarray(uv, dtype=np.float64),
                "point3d_id": point3d_id,
            })

            tracks_by_point[pt_idx].append({
                "image_id": int(image_id),
                "point2d_idx": int(point2d_idx),
            })

        if not use_tqdm and local_count % 1000 == 0:
            log(
                f"  {local_count}/{len(selected_indices)} points terrain reprojetés...",
                1,
                verbose,
            )

    for track in tracks_by_point:
        track.sort(key=lambda x: (x["image_id"], x["point2d_idx"]))

    for fr in frames:
        image_id = fr["image_id"]
        log(
            f"  Image {image_id}: {len(observations_by_image[image_id])} observations",
            1,
            verbose,
        )

    tracked_points = sum(1 for tr in tracks_by_point if len(tr) > 0)

    log(
        f"  Points terrain ayant au moins une observation: {tracked_points}",
        1,
        verbose,
    )

    return observations_by_image, tracks_by_point


def filter_points_with_tracks(pts_xyz, pts_rgb, tracks_by_point, observations_by_image, verbose=1, remap=False):
    kept_old_indices = [i for i, tr in enumerate(tracks_by_point) if len(tr) > 0]

    if len(kept_old_indices) == 0:
        pts_xyz_kept = pts_xyz[:0].copy()
        pts_rgb_kept = None if pts_rgb is None else pts_rgb[:0].copy()
        tracks_kept = []
        observations_new = {k: [] for k in observations_by_image}
        old_to_new_point_id = {}
        point3d_ids_kept = np.zeros((0,), dtype=np.int64)
        return pts_xyz_kept, pts_rgb_kept, tracks_kept, observations_new, old_to_new_point_id, point3d_ids_kept

    point3d_ids_kept = np.asarray([old_idx + 1 for old_idx in kept_old_indices], dtype=np.int64)

    if remap:
        old_to_new_point_id = {old_idx + 1: new_id for new_id, old_idx in enumerate(kept_old_indices, start=1)}
    else:
        old_to_new_point_id = {old_idx + 1: old_idx + 1 for old_idx in kept_old_indices}

    pts_xyz_kept = pts_xyz[kept_old_indices]
    pts_rgb_kept = None if pts_rgb is None else pts_rgb[kept_old_indices]
    tracks_kept = [tracks_by_point[i] for i in kept_old_indices]

    if remap:
        n_old = len(tracks_by_point)
        lut = np.zeros(n_old + 1, dtype=np.int64)
        for new_pid, old_idx in enumerate(kept_old_indices, start=1):
            lut[old_idx + 1] = new_pid

        observations_new = {}
        items = observations_by_image.items()
        use_tqdm = verbose >= 1 and "tqdm" in globals() and tqdm is not None
        if use_tqdm:
            items = tqdm(
                items,
                total=len(observations_by_image),
                desc="Remap observations",
                unit="img",
            )

        for image_id, obs_list in items:
            new_obs_list = []
            append = new_obs_list.append

            for obs in obs_list:
                old_pid = obs["point3d_id"]
                if 0 < old_pid <= n_old:
                    new_pid = int(lut[old_pid])
                    if new_pid != 0:
                        append({
                            "xy": obs["xy"],
                            "point3d_id": new_pid,
                        })

            observations_new[image_id] = new_obs_list

        point3d_ids_kept = np.arange(1, len(kept_old_indices) + 1, dtype=np.int64)

    else:
        observations_new = observations_by_image

    return pts_xyz_kept, pts_rgb_kept, tracks_kept, observations_new, old_to_new_point_id, point3d_ids_kept


def filter_frames_with_observations_and_remap(frames, observations_by_image, tracks_by_point, verbose=1):
    use_tqdm = verbose >= 1 and "tqdm" in globals() and tqdm is not None

    kept_frames = []
    old_to_new_image_id = {}

    for fr in frames:
        old_id = fr["image_id"]
        if not observations_by_image.get(old_id):
            continue

        new_id = len(kept_frames) + 1
        new_fr = dict(fr)
        new_fr["image_id"] = new_id
        kept_frames.append(new_fr)
        old_to_new_image_id[old_id] = new_id

    if not kept_frames:
        log(
            f"  Images conservées après filtrage par observations: 0/{len(frames)}",
            1,
            verbose,
        )
        return [], {}, [[] for _ in range(len(tracks_by_point))], {}

    new_observations_by_image = {}
    for old_id, new_id in old_to_new_image_id.items():
        obs_list = observations_by_image.get(old_id, [])
        new_observations_by_image[new_id] = obs_list

    max_old_image_id = max(fr["image_id"] for fr in frames) if frames else 0
    image_lut = np.zeros(max_old_image_id + 1, dtype=np.int32)
    for old_id, new_id in old_to_new_image_id.items():
        image_lut[int(old_id)] = int(new_id)

    new_tracks_by_point = [None] * len(tracks_by_point)

    iterable = enumerate(tracks_by_point)
    if use_tqdm:
        iterable = tqdm(
            iterable,
            total=len(tracks_by_point),
            desc="Remap tracks",
            unit="pt",
        )

    lut_len = len(image_lut)
    for i, track in iterable:
        if not track:
            new_tracks_by_point[i] = []
            continue

        remapped_track = []
        append = remapped_track.append

        for tr in track:
            old_image_id = int(tr["image_id"])
            if 0 < old_image_id < lut_len:
                new_image_id = int(image_lut[old_image_id])
                if new_image_id != 0:
                    append({
                        "image_id": new_image_id,
                        "point2d_idx": int(tr["point2d_idx"]),
                    })

        if len(remapped_track) > 1:
            remapped_track.sort(key=lambda x: (x["image_id"], x["point2d_idx"]))

        new_tracks_by_point[i] = remapped_track

    log(
        f"  Images conservées après filtrage par observations: {len(kept_frames)}/{len(frames)}",
        1,
        verbose,
    )

    return kept_frames, new_observations_by_image, new_tracks_by_point, old_to_new_image_id


def write_cameras_txt_single_camera(path: Path, width: int, height: int, fx: float, fy: float, cx: float, cy: float):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fx:.12f} {fy:.12f} {cx:.12f} {cy:.12f}\n")


def write_images_txt(path: Path, frames, observations_by_image, verbose: int = 1):
    use_tqdm = verbose >= 1 and "tqdm" in globals() and tqdm is not None
    iterable = frames

    if use_tqdm:
        iterable = tqdm(
            frames,
            total=len(frames),
            desc="Écriture images.txt",
            unit="img",
        )

    with open(path, "w", encoding="utf-8") as f:
        write = f.write

        write("# Image list with two lines of data per image:\n")
        write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        write(f"# Number of images: {len(frames)}\n")

        for fr in iterable:
            q = fr["qvec"]
            t = fr["tvec"]
            image_id = fr["image_id"]
            camera_id = fr["camera_id"]

            write(
                f"{image_id} "
                f"{q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f} "
                f"{t[0]:.12f} {t[1]:.12f} {t[2]:.12f} "
                f"{camera_id} {fr['frame_name']}\n"
            )

            obs = observations_by_image.get(image_id, [])
            write(" ".join(
                f"{o['xy'][0]:.6f} {o['xy'][1]:.6f} {o['point3d_id']}"
                for o in obs
            ))
            write("\n")


def write_points3D_txt(path: Path, pts_xyz, pts_rgb=None, tracks_by_point=None, point3d_ids=None, verbose=1):
    if pts_rgb is None:
        pts_rgb = np.full((len(pts_xyz), 3), 200, dtype=np.uint8)

    if tracks_by_point is None:
        tracks_by_point = [[] for _ in range(len(pts_xyz))]

    if point3d_ids is None:
        point3d_ids = np.arange(1, len(pts_xyz) + 1, dtype=np.int64)

    if len(point3d_ids) != len(pts_xyz):
        raise ValueError(
            f"point3d_ids et pts_xyz doivent avoir la même longueur "
            f"(ids={len(point3d_ids)} vs pts={len(pts_xyz)})"
        )

    use_tqdm = verbose >= 1 and "tqdm" in globals() and tqdm is not None
    iterable = zip(point3d_ids, pts_xyz, pts_rgb, tracks_by_point)

    if use_tqdm:
        iterable = tqdm(
            iterable,
            total=len(pts_xyz),
            desc="Écriture points3D.txt",
            unit="pt",
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(pts_xyz)}\n")

        for pid, p, c, track in iterable:
            parts = [
                str(int(pid)),
                f"{p[0]:.12f}",
                f"{p[1]:.12f}",
                f"{p[2]:.12f}",
                str(int(c[0])),
                str(int(c[1])),
                str(int(c[2])),
                "0.0",
            ]

            for tr in track:
                parts.append(str(tr["image_id"]))
                parts.append(str(tr["point2d_idx"]))

            f.write(" ".join(parts) + "\n")


def apply_transform_to_points(xyz: np.ndarray, T4: np.ndarray, scale: float):
    if len(xyz) == 0:
        return xyz.copy()

    xyz_h = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float64)], axis=1)
    xyz_t = (T4 @ xyz_h.T).T[:, :3]
    xyz_t *= float(scale)
    return xyz_t


def build_transforms_json(path: Path, frames, applied_transform=None, applied_scale=None):
    if not frames:
        raise ValueError("Aucune frame pour transforms.json")

    first = frames[0]
    data = {
        "w": first["width"],
        "h": first["height"],
        "fl_x": first["fx"],
        "fl_y": first["fy"],
        "cx": first["cx"],
        "cy": first["cy"],
        "camera_model": "PINHOLE",
        "ply_file_path": "sparse_pc.ply",
        "frames": []
    }

    applied_transform_4x4 = None
    if applied_transform is not None:
        applied_transform = np.asarray(applied_transform, dtype=np.float64)
        if applied_transform.shape == (4, 4):
            applied_transform_4x4 = applied_transform
            data["applied_transform"] = applied_transform[:3, :4].tolist()
        elif applied_transform.shape == (3, 4):
            applied_transform_4x4 = np.eye(4, dtype=np.float64)
            applied_transform_4x4[:3, :4] = applied_transform
            data["applied_transform"] = applied_transform.tolist()
        else:
            raise ValueError(
                f"applied_transform doit être 3x4 ou 4x4, reçu {applied_transform.shape}"
            )

    if applied_scale is not None:
        data["applied_scale"] = float(applied_scale)

    for fr in frames:
        R_cw = np.asarray(fr["R_cw"], dtype=np.float64)
        t_cw = np.asarray(fr["tvec"], dtype=np.float64).reshape(3, 1)

        w2c = np.concatenate([R_cw, t_cw], axis=1)
        w2c = np.concatenate([w2c, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)], axis=0)

        c2w = np.linalg.inv(w2c)
        c2w[0:3, 1:3] *= -1

        if applied_transform_4x4 is not None:
            c2w = applied_transform_4x4 @ c2w

        if applied_scale is not None:
            c2w = c2w.copy()
            c2w[:3, 3] *= float(applied_scale)

        data["frames"].append({
            "file_path": f'./images/{fr["frame_name"]}',
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": fr["image_id"],
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_scene_normalization_json(path: Path, T4: np.ndarray, scale: float, centers: np.ndarray):
    centroid = centers.mean(axis=0) if len(centers) > 0 else np.zeros(3, dtype=np.float64)

    data = {
        "origin_world": centroid.tolist(),
        "scale": float(scale),
        "translation_matrix_4x4": T4.tolist(),
        "formula_points": "X_normalized = scale * (T @ [X_world, 1])[:3]",
        "formula_camera_centers": "C_normalized = scale * (T @ [C_world, 1])[:3]",
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def parse_images_txt_for_validation(path: Path, verbose: int = 1):
    images = {}

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    record_starts = []
    i = 0
    nlines = len(lines)
    while i < nlines:
        line1 = lines[i].strip()
        if line1 and not line1.startswith("#"):
            parts = line1.split()
            if len(parts) >= 10:
                record_starts.append(i)
                i += 2
                continue
        i += 1

    use_tqdm = verbose >= 1 and "tqdm" in globals() and tqdm is not None
    iterable = record_starts
    if use_tqdm:
        iterable = tqdm(
            record_starts,
            total=len(record_starts),
            desc="Lecture images.txt",
            unit="img",
        )

    for i in iterable:
        parts = lines[i].strip().split()

        image_id = int(parts[0])
        name = parts[9]

        obs = []
        if i + 1 < nlines:
            line2 = lines[i + 1].strip()
            if line2 and not line2.startswith("#"):
                vals = line2.split()
                if len(vals) % 3 != 0:
                    raise ValueError(
                        f"images.txt invalide: image_id={image_id}, "
                        f"la ligne POINTS2D ne contient pas un multiple de 3 valeurs"
                    )

                nobs = len(vals) // 3
                obs = [
                    {
                        "xy": (float(vals[3 * k]), float(vals[3 * k + 1])),
                        "point3d_id": int(vals[3 * k + 2]),
                        "point2d_idx": k,
                    }
                    for k in range(nobs)
                ]

        images[image_id] = {
            "name": name,
            "observations": obs,
        }

    return images


def parse_points3d_txt_for_validation(path: Path):
    points = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 8:
                raise ValueError(
                    f"points3D.txt invalide ligne {line_no}: moins de 8 champs"
                )

            point3d_id = int(parts[0])

            track_parts = parts[8:]
            if len(track_parts) % 2 != 0:
                raise ValueError(
                    f"points3D.txt invalide ligne {line_no}: "
                    f"TRACK[] n'a pas un nombre pair d'éléments"
                )

            track = []
            for i in range(0, len(track_parts), 2):
                image_id = int(track_parts[i])
                point2d_idx = int(track_parts[i + 1])
                track.append((image_id, point2d_idx))

            points[point3d_id] = {
                "track": track
            }

    return points


def try_write_colmap_bin(sparse_dir: Path, verbose: int):
    images_txt = sparse_dir / "images.txt"
    points3d_txt = sparse_dir / "points3D.txt"
    cameras_txt = sparse_dir / "cameras.txt"

    if not images_txt.exists() or not points3d_txt.exists() or not cameras_txt.exists():
        log("[WARN] Fichiers texte COLMAP incomplets: conversion binaire ignorée.", 2, verbose)
        return False

    ok, errors, warnings = validate_colmap_text_model(
        images_txt=images_txt,
        points3d_txt=points3d_txt,
        verbose=verbose,
    )

    if not ok:
        log("[WARN] Le modèle texte est incohérent: conversion .bin annulée.", 1, verbose)
        return False

    try:
        import pycolmap
    except ImportError:
        log("[INFO] pycolmap non installé: génération .txt uniquement.", 1, verbose)
        return False

    try:
        recon = pycolmap.Reconstruction()
        recon.read_text(str(sparse_dir))
        recon.write_binary(str(sparse_dir))
        log("[INFO] Fichiers .bin générés via pycolmap.", 1, verbose)
        return True
    except Exception as e:
        log(f"[WARN] Impossible de générer les .bin avec pycolmap: {e}", 1, verbose)
        return False


def validate_colmap_text_model(images_txt: Path, points3d_txt: Path, verbose: int = 1):
    images = parse_images_txt_for_validation(images_txt, verbose=verbose)
    points = parse_points3d_txt_for_validation(points3d_txt)

    errors = []
    warnings = []

    image_obs_lookup = {}
    for image_id, im in images.items():
        image_obs_lookup[image_id] = im["observations"]

    for point3d_id, pdata in points.items():
        track = pdata["track"]

        for image_id, point2d_idx in track:
            if image_id not in images:
                errors.append(
                    f"Point3D {point3d_id}: image_id {image_id} absent de images.txt"
                )
                continue

            obs = image_obs_lookup[image_id]

            if point2d_idx < 0 or point2d_idx >= len(obs):
                errors.append(
                    f"Point3D {point3d_id}: point2d_idx {point2d_idx} invalide "
                    f"pour image_id {image_id} (taille={len(obs)})"
                )
                continue

            linked_pid = obs[point2d_idx]["point3d_id"]
            if linked_pid != point3d_id:
                errors.append(
                    f"Incohérence track: Point3D {point3d_id} -> "
                    f"(image {image_id}, idx {point2d_idx}) "
                    f"mais images.txt référence {linked_pid}"
                )

    for image_id, im in images.items():
        for o in im["observations"]:
            pid = o["point3d_id"]
            idx = o["point2d_idx"]

            if pid == -1:
                continue

            if pid not in points:
                errors.append(
                    f"Observation orpheline: image {image_id}, idx {idx} -> "
                    f"point3D_id {pid} absent de points3D.txt"
                )
                continue

            track = points[pid]["track"]
            if (image_id, idx) not in track:
                errors.append(
                    f"Incohérence inverse: image {image_id}, idx {idx} -> point {pid}, "
                    f"mais le track du point ne contient pas cette paire"
                )

    points_without_track = [pid for pid, pdata in points.items() if len(pdata["track"]) == 0]
    if points_without_track:
        warnings.append(f"{len(points_without_track)} point(s) 3D sans track")

    image_obs_count = sum(len(im["observations"]) for im in images.values())
    tracked_image_obs_count = sum(
        1 for im in images.values() for o in im["observations"] if o["point3d_id"] != -1
    )

    if verbose >= 1:
        log(
            f"[VALIDATION] images={len(images)}, points3D={len(points)}, "
            f"obs_total={image_obs_count}, obs_trackees={tracked_image_obs_count}",
            1,
            verbose,
        )

    for w in warnings:
        log(f"[VALIDATION][WARN] {w}", 1, verbose)

    if errors:
        for e in errors[:50]:
            log(f"[VALIDATION][ERROR] {e}", 1, verbose)

        if len(errors) > 50:
            log(f"[VALIDATION][ERROR] ... {len(errors) - 50} erreur(s) supplémentaire(s)", 1, verbose)

        return False, errors, warnings

    log("[VALIDATION] Modèle texte COLMAP cohérent.", 1, verbose)
    return True, errors, warnings


def main():
    ap = argparse.ArgumentParser(
        description="Convertit un dossier images + fichiers .CON + LAZ vers une structure de sortie type COLMAP / Nerfstudio."
    )
    ap.add_argument("--laz", required=True, help="Fichier .LAZ")
    ap.add_argument("--image-dir", required=True, help="Dossier racine des images source (scan récursif)")
    ap.add_argument("--out", required=True, help="Dossier de sortie")
    ap.add_argument("--subsample", type=int, default=10, help="Facteur de sous-échantillonnage initial du LAZ")
    ap.add_argument("--image-factor", type=float, default=1.0,
                    help="Facteur de sous-échantillonnage des images (2 = largeur/hauteur divisées par 2)")
    ap.add_argument("--jpeg-quality", type=int, default=95, help="Qualité JPEG de sortie")
    ap.add_argument("--num-terrain-points", type=int, default=5000,
                    help="Nombre de points terrain à reprojeter dans toutes les images")
    ap.add_argument("--xmin", type=float, default=None, help="Borne minimale X optionnelle pour filtrer le LAZ")
    ap.add_argument("--xmax", type=float, default=None, help="Borne maximale X optionnelle pour filtrer le LAZ")
    ap.add_argument("--ymin", type=float, default=None, help="Borne minimale Y optionnelle pour filtrer le LAZ")
    ap.add_argument("--ymax", type=float, default=None, help="Borne maximale Y optionnelle pour filtrer le LAZ")
    ap.add_argument("--remap", action="store_true",
                    help="Remappe les POINT3D_ID en [1..N_kept]. Par défaut désactivé (IDs originaux conservés).")
    ap.add_argument("--z-negative", action="store_true",
                    help="Utilise la convention profondeur z<0 (par défaut: z>0).")
    ap.add_argument("--with-occlusion", action="store_true",
                    help="Active le filtrage d'occlusion (z-buffer local). Désactivé par défaut.")
    ap.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
                    help="0=silencieux, 1=info, 2=warn+info")
    args = ap.parse_args()

    t0 = time.perf_counter()
    dt_start = datetime.now()
    log(f"[TIME] Début: {dt_start.strftime('%Y-%m-%d %H:%M:%S')}", 1, args.verbose)

    laz_path = Path(args.laz)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out)

    script_dir = Path(__file__).resolve().parent
    image_convertor_script = script_dir / "image_convertor.py"

    out_images = out_dir / "images"
    out_colmap = out_dir / "colmap"
    out_sparse = out_colmap / "sparse" / "0"
    out_models_0 = out_sparse / "models" / "0"
    sparse_pc_ply = out_dir / "sparse_pc.ply"
    normalization_json = out_dir / "scene_normalization.json"

    z_positive = not args.z_negative
    log(f"[CONFIG] z_positive={z_positive}", 1, args.verbose)
    log(f"[CONFIG] with_occlusion={args.with_occlusion}", 1, args.verbose)

    for d in [out_images, out_sparse, out_models_0]:
        ensure_dir(d)

    log("[1/8] Indexation récursive des images existantes...", 1, args.verbose)
    image_index_by_stem, image_index_by_name = build_image_index(image_dir)
    log(f"  {len(image_index_by_stem)} images indexées.", 1, args.verbose)

    if not image_index_by_stem:
        print("Aucune image compatible trouvée dans le dossier fourni.")
        sys.exit(2)

    factor = max(float(args.image_factor), 1.0)

    log("[2/8] Lecture des fichiers .CON...", 1, args.verbose)
    frames = []
    intrinsics_ref = None
    skipped_no_con = 0

    for stem, src_img in sorted(image_index_by_stem.items()):
        con_path = resolve_con_path_for_image(src_img, image_dir)

        if con_path is None:
            log(f"[WARN] .CON introuvable pour {src_img}", 2, args.verbose)
            skipped_no_con += 1
            continue

        try:
            ori = parse_con_orientation(con_path)
        except Exception as e:
            log(f"[WARN] Impossible de lire {con_path.name}: {e}", 2, args.verbose)
            continue

        width, height, fx, fy, cx, cy = scale_intrinsics(
            ori["width"], ori["height"], ori["fx"], ori["fy"], ori["cx"], ori["cy"], factor
        )
        transfo2d_scaled = scale_transfo2d(ori["transfo2d"], factor)

        if intrinsics_ref is None:
            intrinsics_ref = (width, height, fx, fy, cx, cy)
            log(f"  Intrinsics de référence: {intrinsics_ref}", 1, args.verbose)
        else:
            ref = intrinsics_ref
            cur = (width, height, fx, fy, cx, cy)
            same = (
                ref[0] == cur[0] and
                ref[1] == cur[1] and
                abs(ref[2] - cur[2]) < 1e-6 and
                abs(ref[3] - cur[3]) < 1e-6 and
                abs(ref[4] - cur[4]) < 1e-6 and
                abs(ref[5] - cur[5]) < 1e-6
            )
            if not same:
                raise ValueError(
                    "Le script actuel n'exporte qu'une seule caméra COLMAP. "
                    f"Intrinsics différentes détectées pour {src_img.name}: "
                    f"ref={ref}, cur={cur}"
                )

        frames.append({
            "image_id": len(frames) + 1,
            "camera_id": 1,
            "frame_name": None,
            "source_stem": stem,
            "source_image": str(src_img),
            "con_path": str(con_path),
            "center": ori["center"],
            "qvec": ori["qvec"],
            "tvec": ori["tvec"],
            "R_cw": ori["R_cw"],
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "transfo2d": transfo2d_scaled,
        })

    log(f"  Total images avec .CON retenues: {len(frames)}", 1, args.verbose)
    if skipped_no_con > 0:
        log(f"  Images ignorées faute de .CON: {skipped_no_con}", 1, args.verbose)

    if not frames:
        print("Aucune image exploitable avec fichier .CON trouvé.")
        sys.exit(3)

    width, height, fx, fy, cx, cy = intrinsics_ref

    log("[3/8] Lecture et sous-échantillonnage du LAZ...", 1, args.verbose)
    pts_xyz_raw, pts_rgb = read_laz_points(laz_path, args.subsample)
    log(f"  {len(pts_xyz_raw)} points conservés après sous-échantillonnage initial.", 1, args.verbose)

    bbox_enabled = any(v is not None for v in (args.xmin, args.xmax, args.ymin, args.ymax))
    if bbox_enabled:
        log("[3b/8] Filtrage du LAZ par bbox XY...", 1, args.verbose)
        before_bbox = len(pts_xyz_raw)
        pts_xyz_raw, pts_rgb = filter_xy_bbox(
            pts_xyz_raw, pts_rgb,
            xmin=args.xmin, xmax=args.xmax,
            ymin=args.ymin, ymax=args.ymax,
        )
        log(f"  {len(pts_xyz_raw)}/{before_bbox} points conservés dans la bbox.", 1, args.verbose)

    if len(pts_xyz_raw) == 0:
        print("Aucun point LAZ conservé après filtrage.")
        sys.exit(4)

    log("[4/8] Génération des observations synthétiques (repère COLMAP source)...", 1, args.verbose)
    observations_by_image, tracks_by_point = build_synthetic_observations(
        frames=frames,
        pts_xyz=pts_xyz_raw,
        pts_rgb=pts_rgb,
        num_terrain_points=args.num_terrain_points,
        z_positive=z_positive,
        verbose=args.verbose,
        with_occlusion=args.with_occlusion,
    )

    pts_xyz_kept, pts_rgb_kept, tracks_kept, observations_by_image, old_to_new_point_id, point3d_ids_kept = filter_points_with_tracks(
        pts_xyz=pts_xyz_raw,
        pts_rgb=pts_rgb,
        tracks_by_point=tracks_by_point,
        observations_by_image=observations_by_image,
        verbose=args.verbose,
        remap=args.remap,
    )

    log(f"  Points 3D exportés avec tracks: {len(pts_xyz_kept)}", 1, args.verbose)

    if len(pts_xyz_kept) == 0:
        print("Aucun point 3D avec track après filtrage.")
        sys.exit(5)

    log("[5/8] Filtrage des images sans homologue conservé...", 1, args.verbose)
    frames, observations_by_image, tracks_kept, old_to_new_image_id = filter_frames_with_observations_and_remap(
        frames=frames,
        observations_by_image=observations_by_image,
        tracks_by_point=tracks_kept,
        verbose=args.verbose,
    )

    if not frames:
        print("Aucune image ne possède de point homologue conservé.")
        sys.exit(6)

    log("[6/8] Préparation des images utiles uniquement (conversion / copie / sous-échantillonnage)...", 1, args.verbose)
    selected_source_paths = [Path(fr["source_image"]) for fr in frames]
    exported_index = prepare_selected_images(
        source_paths=selected_source_paths,
        output_dir=out_images,
        factor=factor,
        jpeg_quality=args.jpeg_quality,
        verbose=args.verbose,
        image_convertor_script=image_convertor_script,
    )

    for fr in frames:
        stem = fr["source_stem"]
        exported_img = exported_index.get(stem)
        if exported_img is None:
            raise RuntimeError(f"Image exportée absente après préparation: {stem}")
        fr["frame_name"] = exported_img.name

    log(f"  {len(exported_index)} image(s) utile(s) préparée(s).", 1, args.verbose)

    log("[7/8] Calcul de la normalisation Nerfstudio (transforms + PLY uniquement)...", 1, args.verbose)
    centers = np.stack([fr["center"] for fr in frames], axis=0)
    applied_transform = get_nerfstudio_axis_transform_4x4()
    applied_scale = 1.0

    pts_xyz_kept_ns = apply_transform_to_points(pts_xyz_kept, applied_transform, applied_scale)

    write_scene_normalization_json(
        normalization_json,
        applied_transform,
        applied_scale,
        centers,
    )

    log("[8/8] Écriture COLMAP, sparse_pc.ply, transforms.json + conversion binaire...", 1, args.verbose)
    write_cameras_txt_single_camera(out_sparse / "cameras.txt", width, height, fx, fy, cx, cy)
    write_images_txt(out_sparse / "images.txt", frames, observations_by_image, verbose=args.verbose)
    write_points3D_txt(
        out_sparse / "points3D.txt",
        pts_xyz_kept,
        pts_rgb_kept,
        tracks_kept,
        point3d_ids=point3d_ids_kept,
        verbose=args.verbose,
    )
    write_ply_xyzrgb(sparse_pc_ply, pts_xyz_kept_ns, pts_rgb_kept, verbose=args.verbose)

    build_transforms_json(
        out_dir / "transforms.json",
        frames,
        applied_transform=applied_transform,
        applied_scale=applied_scale,
    )

    bin_ok = try_write_colmap_bin(out_sparse, args.verbose)

    if bin_ok:
        for txt_name in ("cameras.txt", "images.txt", "points3D.txt"):
            txt_path = out_sparse / txt_name
            try:
                txt_path.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                log(f"[WARN] Impossible de supprimer {txt_path}: {e}", 1, args.verbose)

    print("\nTerminé.")
    print(f"Sortie: {out_dir}")
    print(f"Images utiles: {out_images}")
    print(f"COLMAP sparse: {out_sparse}")
    print(f"Sparse PLY Nerfstudio: {sparse_pc_ply}")
    print(f"Transforms Nerfstudio: {out_dir / 'transforms.json'}")
    print(f"Normalisation: {normalization_json}")

    dt_end = datetime.now()
    elapsed_s = time.perf_counter() - t0

    h = int(elapsed_s // 3600)
    m = int((elapsed_s % 3600) // 60)
    s = elapsed_s % 60

    print(f"Heure début: {dt_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Heure fin  : {dt_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Durée totale: {h:02d}:{m:02d}:{int(s):02d}")


if __name__ == "__main__":
    main()

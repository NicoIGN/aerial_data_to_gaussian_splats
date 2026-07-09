#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import struct
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ExifTags
import subprocess


# =============================================================================
# Maths / rotations
# =============================================================================

def qvec_to_rotmat(qvec):
    qvec = np.asarray(qvec, dtype=np.float64)
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z,     2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z,     1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y,     2 * y * z + 2 * w * x,     1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


# =============================================================================
# Lecture COLMAP
# =============================================================================

def load_colmap_images(colmap_dir: Path):
    bin_candidates = [
        colmap_dir / "sparse" / "0" / "images.bin",
        colmap_dir / "images.bin",
    ]
    txt_candidates = [
        colmap_dir / "sparse" / "0" / "images.txt",
        colmap_dir / "images.txt",
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
                    "qvec": np.array(qvec, dtype=np.float64),
                    "tvec": np.array(tvec, dtype=np.float64),
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
            qvec = np.array(
                [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])],
                dtype=np.float64,
            )
            tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])], dtype=np.float64)
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

    for p in bin_candidates:
        if p.exists():
            return _read_images_bin(p)

    for p in txt_candidates:
        if p.exists():
            return _read_images_txt(p)

    raise FileNotFoundError("images.bin/images.txt introuvable")


def load_colmap_cameras(colmap_dir: Path):
    bin_candidates = [
        colmap_dir / "sparse" / "0" / "cameras.bin",
        colmap_dir / "cameras.bin",
    ]
    txt_candidates = [
        colmap_dir / "sparse" / "0" / "cameras.txt",
        colmap_dir / "cameras.txt",
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
                    0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8,
                    6: 12, 7: 5, 8: 4, 9: 5, 10: 12,
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

    for p in bin_candidates:
        if p.exists():
            return _read_bin(p)

    for p in txt_candidates:
        if p.exists():
            return _read_txt(p)

    raise FileNotFoundError("cameras.bin/cameras.txt introuvable")


def camera_to_intrinsics(cam):
    model = cam["model"]
    p = cam["params"]

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p[:3]
        fx = fy = f
        k1 = k2 = p1 = p2 = 0.0
    elif model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
        k1 = k2 = p1 = p2 = 0.0
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = p[:4]
        fx = fy = f
        k2 = p1 = p2 = 0.0
    elif model == "RADIAL":
        f, cx, cy, k1, k2 = p[:5]
        fx = fy = f
        p1 = p2 = 0.0
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p[:8]
    else:
        raise NotImplementedError(f"Modèle non géré: {model}")

    return {
        "width": cam["width"],
        "height": cam["height"],
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "k1": float(k1),
        "k2": float(k2),
        "p1": float(p1),
        "p2": float(p2),
    }


# =============================================================================
# Distorsion COLMAP
# =============================================================================

def distort_colmap_points(u, v, intr):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    fx = intr["fx"]
    fy = intr["fy"]
    cx = intr["cx"]
    cy = intr["cy"]
    k1 = intr["k1"]
    k2 = intr["k2"]
    p1 = intr["p1"]
    p2 = intr["p2"]

    x = (u - cx) / fx
    y = (v - cy) / fy
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2

    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    ud = cx + fx * xd
    vd = cy + fy * yd
    return ud, vd


# =============================================================================
# Modèle .CON commun
# =============================================================================

def con_apply_distortion(u0, v0, con_intr):
    u0 = np.asarray(u0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)

    pps_c = con_intr["pps_c"]
    pps_l = con_intr["pps_l"]
    r1 = con_intr["r1"]
    r3 = con_intr["r3"]
    r5 = con_intr.get("r5", 0.0)
    r7 = con_intr.get("r7", 0.0)

    du = u0 - pps_c
    dv = v0 - pps_l
    r2 = du * du + dv * dv

    scale = 1.0 + r1 * r2 + r3 * r2 * r2 + r5 * r2 * r2 * r2 + r7 * r2 * r2 * r2 * r2
    ud = pps_c + du * scale
    vd = pps_l + dv * scale
    return ud, vd


def con_project_world_to_image(con_ori, xyz):
    R_cw = con_ori["R_cw"]
    tvec = con_ori["tvec"]
    f = con_ori["focal"]
    ppa_c = con_ori["ppa_c"]
    ppa_l = con_ori["ppa_l"]

    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
    Xc = R_cw @ xyz + tvec
    xb, yb, zb = Xc

    if zb <= 1e-9:
        return None, Xc

    u0 = ppa_c + f * (xb / zb)
    v0 = ppa_l + f * (yb / zb)
    ud, vd = con_apply_distortion(u0, v0, con_ori)
    return np.array([ud, vd], dtype=np.float64), Xc


def con_map_image_to_image_equivalent(u, v, con_intr, src_intr):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    x = (u - src_intr["cx"]) / src_intr["fx"]
    y = (v - src_intr["cy"]) / src_intr["fy"]

    u0 = con_intr["ppa_c"] + con_intr["focal"] * x
    v0 = con_intr["ppa_l"] + con_intr["focal"] * y
    return con_apply_distortion(u0, v0, con_intr)


# =============================================================================
# Outils de grille / validation
# =============================================================================

def make_image_grid(width, height, step_px=100, include_border=True):
    xs = np.arange(0, width, step_px, dtype=np.float64)
    ys = np.arange(0, height, step_px, dtype=np.float64)

    if include_border:
        if len(xs) == 0 or xs[-1] != width - 1:
            xs = np.unique(np.append(xs, width - 1))
        if len(ys) == 0 or ys[-1] != height - 1:
            ys = np.unique(np.append(ys, height - 1))

    XX, YY = np.meshgrid(xs, ys)
    return XX.ravel(), YY.ravel()


def validate_conic_equivalent(con_intr, src_intr, step_px=100):
    u, v = make_image_grid(src_intr["width"], src_intr["height"], step_px=step_px)

    uc, vc = distort_colmap_points(u, v, src_intr)
    uq, vq = con_map_image_to_image_equivalent(u, v, con_intr, src_intr)

    err = np.sqrt((uq - uc) ** 2 + (vq - vc) ** 2)

    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean": float(np.mean(err)),
        "max": float(np.max(err)),
        "count": int(err.size),
    }


# =============================================================================
# Fit COLMAP -> .CON
# =============================================================================

def fit_conic_equivalent(
    intr,
    grid_step_px=100,
    focal_rel_span=0.08,
    focal_steps=17,
    pps_search_radius_px=80.0,
    pps_step_px=10.0,
):
    """
    Ajuste un modèle .CON équivalent au modèle COLMAP en optimisant :
      - focale équivalente
      - PPS effectif
      - r1, r3
    en gardant PPA=(cx,cy).

    Le fit est fait sur une grille image régulière (typiquement tous les 100 px).
    """
    width = intr["width"]
    height = intr["height"]
    cx = intr["cx"]
    cy = intr["cy"]
    fx = intr["fx"]
    fy = intr["fy"]

    u, v = make_image_grid(width, height, step_px=grid_step_px)
    ud_target, vd_target = distort_colmap_points(u, v, intr)

    x = (u - cx) / fx
    y = (v - cy) / fy

    f0 = 0.5 * (fx + fy)
    focal_scales = np.linspace(1.0 - focal_rel_span, 1.0 + focal_rel_span, focal_steps)
    pps_offsets = np.arange(
        -pps_search_radius_px,
        pps_search_radius_px + 0.5 * pps_step_px,
        pps_step_px,
        dtype=np.float64,
    )

    best = None

    for sf in focal_scales:
        f_eq = f0 * sf

        # pinhole isotrope équivalent
        u0 = cx + f_eq * x
        v0 = cy + f_eq * y

        for dc0 in pps_offsets:
            for dl0 in pps_offsets:
                pps_c = cx + dc0
                pps_l = cy + dl0

                du0 = u0 - pps_c
                dv0 = v0 - pps_l
                R2 = du0 * du0 + dv0 * dv0

                valid = R2 > 1e-12
                if not np.any(valid):
                    continue

                # estimation locale du scale radial
                tu = ud_target - pps_c
                tv = vd_target - pps_l
                scale_obs = np.ones_like(u0)
                scale_obs[valid] = (tu[valid] * du0[valid] + tv[valid] * dv0[valid]) / R2[valid]

                y_ls = scale_obs - 1.0
                A = np.stack([R2, R2 * R2], axis=1)

                A_valid = A[valid]
                y_valid = y_ls[valid]

                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A_valid, y_valid, rcond=None)
                except np.linalg.LinAlgError:
                    continue

                r1, r3 = coeffs.tolist()

                candidate = {
                    "ppa_c": float(cx),
                    "ppa_l": float(cy),
                    "focal": float(f_eq),
                    "pps_c": float(pps_c),
                    "pps_l": float(pps_l),
                    "r1": float(r1),
                    "r3": float(r3),
                    "r5": 0.0,
                    "r7": 0.0,
                }

                uq, vq = con_map_image_to_image_equivalent(u, v, candidate, intr)
                err = np.sqrt((uq - ud_target) ** 2 + (vq - vd_target) ** 2)
                rmse = float(np.sqrt(np.mean(err ** 2)))
                max_err = float(np.max(err))

                candidate["rmse"] = rmse
                candidate["max_err"] = max_err

                if best is None or candidate["rmse"] < best["rmse"]:
                    best = candidate

    if best is None:
        best = {
            "ppa_c": float(cx),
            "ppa_l": float(cy),
            "focal": float(0.5 * (fx + fy)),
            "pps_c": float(cx),
            "pps_l": float(cy),
            "r1": 0.0,
            "r3": 0.0,
            "r5": 0.0,
            "r7": 0.0,
            "rmse": float("inf"),
            "max_err": float("inf"),
        }

    return best


# =============================================================================
# Lecture / écriture .CON
# =============================================================================

def parse_con_orientation(con_path: Path):
    tree = ET.parse(con_path)
    root = tree.getroot()

    geometry = root.find("geometry")
    extr = geometry.find("extrinseque")
    intr = geometry.find("intrinseque")
    sensor = intr.find("sensor")

    eucl = extr.find("systeme/euclidien")
    sommet = extr.find("sommet")
    rotation = extr.find("rotation")

    center = np.array([
        float(eucl.findtext("x")) + float(sommet.findtext("easting")),
        float(eucl.findtext("y")) + float(sommet.findtext("northing")),
        float(sommet.findtext("altitude")),
    ], dtype=np.float64)

    image2ground = str(rotation.findtext("Image2Ground")).strip().lower() == "true"

    def _read_row(row_tag):
        row = rotation.find(f"mat3d/{row_tag}/pt3d")
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

    R_cw = M.T if image2ground else M
    tvec = -R_cw @ center

    ppa_c = float(sensor.findtext("ppa/c"))
    ppa_l = float(sensor.findtext("ppa/l"))
    focal = float(sensor.findtext("ppa/focale"))

    dist = sensor.find("distortion")
    if dist is not None and dist.find("pps") is not None:
        pps_c = float(dist.findtext("pps/c"))
        pps_l = float(dist.findtext("pps/l"))
        r1 = float(dist.findtext("r1", "0"))
        r3 = float(dist.findtext("r3", "0"))
        r5 = float(dist.findtext("r5", "0"))
        r7 = float(dist.findtext("r7", "0"))
    else:
        pps_c = ppa_c
        pps_l = ppa_l
        r1 = r3 = r5 = r7 = 0.0

    return {
        "center_world": center,
        "R_cw": R_cw,
        "tvec": tvec,
        "width": int(sensor.findtext("image_size/width")),
        "height": int(sensor.findtext("image_size/height")),
        "ppa_c": ppa_c,
        "ppa_l": ppa_l,
        "pps_c": pps_c,
        "pps_l": pps_l,
        "focal": focal,
        "r1": r1,
        "r3": r3,
        "r5": r5,
        "r7": r7,
    }


def add_text(parent, tag, value):
    el = ET.SubElement(parent, tag)
    el.text = str(value)
    return el


def add_pt3d(parent, x, y, z):
    pt = ET.SubElement(parent, "pt3d")
    add_text(pt, "x", f"{x:.15f}")
    add_text(pt, "y", f"{y:.15f}")
    add_text(pt, "z", f"{z:.15f}")
    return pt


def prettify_xml(elem):
    rough = ET.tostring(elem, encoding="utf-8")
    from xml.dom import minidom
    return minidom.parseString(rough).toprettyxml(indent="    ", encoding="utf-8")


def build_orientation_xml(image_name, center, R_cw, con_intr, geodesic_name, pixel_size_value):
    root = ET.Element("orientation")
    add_text(root, "version", "1.0")

    auxiliarydata = ET.SubElement(root, "auxiliarydata")
    add_text(auxiliarydata, "image_name", image_name)

    image_date = ET.SubElement(auxiliarydata, "image_date")
    for tag in ["year", "month", "day", "hour", "minute", "second"]:
        add_text(image_date, tag, "0")
    add_text(image_date, "time_system", "")
    ET.SubElement(auxiliarydata, "samples")

    geometry = ET.SubElement(root, "geometry", {"type": "physique"})

    extr = ET.SubElement(geometry, "extrinseque")
    systeme = ET.SubElement(extr, "systeme")

    euclidien = ET.SubElement(systeme, "euclidien", {"type": "MATISRTL"})
    add_text(euclidien, "x", f"{center[0]:.15f}")
    add_text(euclidien, "y", f"{center[1]:.15f}")

    add_text(systeme, "geodesique", geodesic_name)
    add_text(extr, "grid_alti", "UNKNOWN")

    sommet = ET.SubElement(extr, "sommet")
    add_text(sommet, "easting", "0")
    add_text(sommet, "northing", "0")
    add_text(sommet, "altitude", f"{center[2]:.15f}")

    rotation = ET.SubElement(extr, "rotation")
    add_text(rotation, "Image2Ground", "false")

    mat3d = ET.SubElement(rotation, "mat3d")
    for i in range(3):
        li = ET.SubElement(mat3d, f"l{i+1}")
        add_pt3d(li, R_cw[i, 0], R_cw[i, 1], R_cw[i, 2])

    intrinseque = ET.SubElement(geometry, "intrinseque")
    sensor = ET.SubElement(intrinseque, "sensor")
    add_text(sensor, "name", "ESTIMATED_SENSOR")

    calib_date = ET.SubElement(sensor, "calibration_date")
    for tag in ["year", "month", "day", "hour", "minute", "second"]:
        add_text(calib_date, tag, "0")
    add_text(calib_date, "time_system", "")

    add_text(sensor, "serial_number", "UNKNOWN")

    image_size = ET.SubElement(sensor, "image_size")
    add_text(image_size, "width", con_intr["width"])
    add_text(image_size, "height", con_intr["height"])

    sensor_size = ET.SubElement(sensor, "sensor_size")
    add_text(sensor_size, "width", con_intr["width"])
    add_text(sensor_size, "height", con_intr["height"])

    ppa = ET.SubElement(sensor, "ppa")
    add_text(ppa, "c", f"{con_intr['ppa_c']:.15f}")
    add_text(ppa, "l", f"{con_intr['ppa_l']:.15f}")
    add_text(ppa, "focale", f"{con_intr['focal']:.15f}")

    distortion = ET.SubElement(sensor, "distortion")
    pps = ET.SubElement(distortion, "pps")
    add_text(pps, "c", f"{con_intr['pps_c']:.15f}")
    add_text(pps, "l", f"{con_intr['pps_l']:.15f}")

    add_text(distortion, "r1", f"{con_intr['r1']:.15e}")
    add_text(distortion, "r3", f"{con_intr['r3']:.15e}")
    add_text(distortion, "r5", f"{con_intr.get('r5', 0.0):.15e}")
    add_text(distortion, "r7", f"{con_intr.get('r7', 0.0):.15e}")

    add_text(sensor, "pixel_size", pixel_size_value)

    return root


# =============================================================================
# EXIF
# =============================================================================

def _exif_dict(img):
    exif = img.getexif()
    if not exif:
        return {}
    out = {}
    for tag_id, val in exif.items():
        name = ExifTags.TAGS.get(tag_id, tag_id)
        out[name] = val
    return out


def _rational_to_float(v):
    try:
        if hasattr(v, "numerator") and hasattr(v, "denominator"):
            return float(v.numerator) / float(v.denominator)
        if isinstance(v, tuple) and len(v) == 2 and v[1] != 0:
            return float(v[0]) / float(v[1])
        return float(v)
    except Exception:
        return None

# =============================================================================
# EXIF - lecture de la focale image
# =============================================================================

def get_exif_focal_mm_from_pil(image_path: Path):
    """
    Essaie de lire la focale EXIF via Pillow.
    Retourne la focale en millimètres, ou None si absente / illisible.
    """
    try:
        with Image.open(image_path) as img:
            exif = _exif_dict(img)
        if "FocalLength" not in exif:
            return None
        return _rational_to_float(exif["FocalLength"])
    except Exception:
        return None


def _parse_focal_length_string(raw):
    """
    Parse des formes courantes:
      - '21mm'
      - '21.0 mm'
      - '21/1'
      - '21'
    Retourne un float en mm ou None.
    """
    if raw is None:
        return None

    s = str(raw).strip()
    if not s:
        return None

    s = s.replace(" mm", "mm").replace("MM", "mm")

    if s.lower().endswith("mm"):
        s = s[:-2].strip()

    if "/" in s:
        try:
            a, b = s.split("/", 1)
            a = float(a.strip())
            b = float(b.strip())
            if abs(b) < 1e-12:
                return None
            return a / b
        except Exception:
            return None

    try:
        return float(s)
    except Exception:
        return None


def get_exif_focal_mm_from_exiv2(image_path: Path):
    """
    Fallback robuste via exiv2.

    Stratégie:
      1. essaie exiv2 ciblé sur Exif.Photo.FocalLength
      2. sinon essaie la sortie humaine de exiv2
    """
    # -------------------------------------------------------------------------
    # tentative 1: sortie ciblée machine-friendly
    # -------------------------------------------------------------------------
    try:
        cmd = ["exiv2", "-g", "Exif.Photo.FocalLength", "-Pt", str(image_path)]
        res = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]

        for line in lines:
            val = _parse_focal_length_string(line)
            if val is not None:
                return val

            parts = line.split()
            if parts:
                val = _parse_focal_length_string(parts[-1])
                if val is not None:
                    return val
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # tentative 2: sortie humaine classique de exiv2
    # ex:
    #   Focal length    : 21.0 mm
    # -------------------------------------------------------------------------
    try:
        cmd = ["exiv2", str(image_path)]
        res = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )

        for line in res.stdout.splitlines():
            if ":" not in line:
                continue

            left, right = line.split(":", 1)
            key = left.strip().lower()
            value = right.strip()

            if key == "focal length":
                val = _parse_focal_length_string(value)
                if val is not None:
                    return val
    except Exception:
        pass

    return None


def get_exif_focal_mm(image_path: Path):
    """
    Retourne la focale EXIF en mm.
    Ordre de tentative:
      1. PIL
      2. exiv2
    """
    focal_mm = get_exif_focal_mm_from_pil(image_path)
    if focal_mm is not None:
        return focal_mm

    focal_mm = get_exif_focal_mm_from_exiv2(image_path)
    if focal_mm is not None:
        return focal_mm

    return None

# =============================================================================
# FIN EXIF - lecture de la focale image
# =============================================================================


def estimate_pixel_size_from_exif_and_colmap_focal(image_path: Path, fx: float, fy: float):
    focal_mm = get_exif_focal_mm(image_path)
    if focal_mm is None:
        return None

    focal_px = 0.5 * (float(fx) + float(fy))
    if focal_px <= 0:
        return None

    return (focal_mm * 1e-3) / focal_px

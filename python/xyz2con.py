#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    print("Erreur: scipy requis. Installe: pip install scipy")
    sys.exit(1)


def parse_xyz(xyz_path: Path):
    rows = []

    with open(xyz_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue

            parts = s.split()
            if len(parts) < 15:
                raise ValueError(
                    f"{xyz_path}: ligne {line_no} invalide, "
                    f"attendu au moins 15 colonnes, reçu {len(parts)}"
                )

            rows.append({
                "label": parts[0],
                "X0": float(parts[1]),
                "Y0": float(parts[2]),
                "Z0": float(parts[3]),
                "omega_deg": float(parts[4]),
                "phi_deg": float(parts[5]),
                "kappa_deg": float(parts[6]),
                "c": float(parts[7]),
                "x0": float(parts[8]),
                "y0": float(parts[9]),
                "a3": float(parts[10]),
                "a4": float(parts[11]),
                "a5": float(parts[12]),
                "a6": float(parts[13]),
                "rho0": float(parts[14]),
            })

    return rows


def fmt_float(v: float, digits: int = 12):
    return f"{v:.{digits}f}"


def get_convention_matrix(name: str):
    name = (name or "none").lower()

    if name == "none":
        return np.eye(3, dtype=np.float64)

    if name == "swap_xy_cw90":
        return np.array([
            [0.0,  1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0,  0.0, 1.0],
        ], dtype=np.float64)

    if name == "swap_xy_ccw90":
        return np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ], dtype=np.float64)

    if name == "flip_y":
        return np.array([
            [1.0,  0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0,  0.0, 1.0],
        ], dtype=np.float64)

    if name == "flip_x":
        return np.array([
            [-1.0, 0.0, 0.0],
            [0.0,  1.0, 0.0],
            [0.0,  0.0, 1.0],
        ], dtype=np.float64)

    if name == "flip_xy":
        return np.array([
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0,  0.0, 1.0],
        ], dtype=np.float64)

    raise ValueError(f"Convention inconnue: {name}")


def build_rotation_matrix_photogrammetry(
    omega_deg: float,
    phi_deg: float,
    kappa_deg: float,
    *,
    euler_order: str = "xyz",
    kappa_offset_deg: float = 0.0,
    convention: str = "none",
    convention_side: str = "right",
    transpose: bool = False,
):
    base = R.from_euler(
        euler_order,
        [omega_deg, phi_deg, kappa_deg + kappa_offset_deg],
        degrees=True,
    ).as_matrix()

    conv = get_convention_matrix(convention)

    if convention_side == "right":
        M = base @ conv
    elif convention_side == "left":
        M = conv @ base
    else:
        raise ValueError(f"convention_side invalide: {convention_side}")

    if transpose:
        M = M.T

    det = np.linalg.det(M)
    if det < 0:
        raise ValueError(
            f"Matrice de rotation invalide (det<0). "
            f"Vérifie la convention choisie. det={det}"
        )

    return M


def write_con_file(
    out_path: Path,
    image_name: str,
    X0: float,
    Y0: float,
    Z0: float,
    M: np.ndarray,
    width: int,
    height: int,
    focal: float,
    ppa_c: float,
    ppa_l: float,
    geodesique: str,
    euclidien_type: str,
    sensor_name: str,
    pixel_size: float | None,
    image2ground: bool,
):
    root = ET.Element("orientation")

    ET.SubElement(root, "lastmodificationbylibori", attrib={
        "date": "2026-06-22",
        "time": "00 h 00 min 00 sec"
    })
    ET.SubElement(root, "version").text = "1.0"

    auxiliarydata = ET.SubElement(root, "auxiliarydata")
    ET.SubElement(auxiliarydata, "image_name").text = Path(image_name).stem

    image_date = ET.SubElement(auxiliarydata, "image_date")
    for tag in ("year", "month", "day", "hour", "minute", "second"):
        ET.SubElement(image_date, tag).text = "0"
    ET.SubElement(image_date, "time_system").text = ""
    ET.SubElement(auxiliarydata, "samples")

    geometry = ET.SubElement(root, "geometry", attrib={"type": "physique"})

    extr = ET.SubElement(geometry, "extrinseque")
    systeme = ET.SubElement(extr, "systeme")

    euclidien = ET.SubElement(systeme, "euclidien", attrib={"type": euclidien_type})
    ET.SubElement(euclidien, "x").text = fmt_float(X0)
    ET.SubElement(euclidien, "y").text = fmt_float(Y0)

    ET.SubElement(systeme, "geodesique").text = geodesique
    ET.SubElement(extr, "grid_alti").text = "UNKNOWN"

    sommet = ET.SubElement(extr, "sommet")
    ET.SubElement(sommet, "easting").text = "0"
    ET.SubElement(sommet, "northing").text = "0"
    ET.SubElement(sommet, "altitude").text = fmt_float(Z0)

    rotation = ET.SubElement(extr, "rotation")
    ET.SubElement(rotation, "Image2Ground").text = "true" if image2ground else "false"

    mat3d = ET.SubElement(rotation, "mat3d")
    for row_name, row in zip(("l1", "l2", "l3"), M):
        l = ET.SubElement(mat3d, row_name)
        pt3d = ET.SubElement(l, "pt3d")
        ET.SubElement(pt3d, "x").text = f"{row[0]:.15f}"
        ET.SubElement(pt3d, "y").text = f"{row[1]:.15f}"
        ET.SubElement(pt3d, "z").text = f"{row[2]:.15f}"

    intr = ET.SubElement(geometry, "intrinseque")
    sensor = ET.SubElement(intr, "sensor")

    ET.SubElement(sensor, "name").text = sensor_name

    calibration_date = ET.SubElement(sensor, "calibration_date")
    for tag in ("year", "month", "day", "hour", "minute", "second"):
        ET.SubElement(calibration_date, tag).text = "0"
    ET.SubElement(calibration_date, "time_system").text = ""

    ET.SubElement(sensor, "serial_number").text = "UNKNOWN"

    image_size = ET.SubElement(sensor, "image_size")
    ET.SubElement(image_size, "width").text = str(width)
    ET.SubElement(image_size, "height").text = str(height)

    sensor_size = ET.SubElement(sensor, "sensor_size")
    ET.SubElement(sensor_size, "width").text = str(width)
    ET.SubElement(sensor_size, "height").text = str(height)

    ppa = ET.SubElement(sensor, "ppa")
    ET.SubElement(ppa, "c").text = fmt_float(ppa_c)
    ET.SubElement(ppa, "l").text = fmt_float(ppa_l)
    ET.SubElement(ppa, "focale").text = fmt_float(focal)

    if pixel_size is not None:
        ET.SubElement(sensor, "pixel_size").text = f"{pixel_size:.12g}"

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ", level=0)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def main():
    ap = argparse.ArgumentParser(
        description="Convertit un fichier .xyz photogrammétrique OPK en fichiers .CON (un par image)."
    )
    ap.add_argument("--xyz", required=True, help="Fichier .xyz d'entrée")
    ap.add_argument("--out-dir", required=True, help="Dossier de sortie des fichiers .CON")

    ap.add_argument("--width", type=int, required=True, help="Largeur image")
    ap.add_argument("--height", type=int, required=True, help="Hauteur image")

    ap.add_argument("--focal", type=float, default=None,
                    help="Focale à écrire dans le .CON. Par défaut: valeur c du .xyz")

    ap.add_argument("--ppa-c", type=float, default=None,
                    help="PPA colonne explicite")
    ap.add_argument("--ppa-l", type=float, default=None,
                    help="PPA ligne explicite")
    ap.add_argument("--use-center-ppa", action="store_true",
                    help="Utilise width/2 et height/2 au lieu de x0/-y0 du .xyz")

    ap.add_argument("--pixel-size", type=float, default=None, help="Pixel size optionnel")
    ap.add_argument("--geodesique", default="LAMBERT93", help="Nom du système géodésique")
    ap.add_argument("--euclidien-type", default="MATISRTL", help="Type de la balise euclidien")
    ap.add_argument("--sensor-name", default="DRONE_CAMERA", help="Nom du capteur")

    ap.add_argument("--euler-order", default="xyz",
                    choices=["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"],
                    help="Ordre Euler utilisé comme base pour omega, phi, kappa")
    ap.add_argument("--kappa-offset", type=float, default=0.0,
                    help="Offset ajouté à kappa en degrés. Par défaut: 0 (ancienne version)")
    ap.add_argument("--convention", default="none",
                    choices=["none", "swap_xy_cw90", "swap_xy_ccw90", "flip_y", "flip_x", "flip_xy"],
                    help="Matrice fixe de convention image/caméra à appliquer")
    ap.add_argument("--convention-side", default="right",
                    choices=["left", "right"],
                    help="Applique la convention à gauche ou à droite de la rotation de base")
    ap.add_argument("--flip-image-180", action="store_true",
                    help="Applique un retournement de 180° dans le plan image (équivaut à convention flip_xy)")
    ap.add_argument("--transpose", action="store_true",
                    help="Transpose la matrice finale avant écriture")
    ap.add_argument("--image2ground", action="store_true",
                    help="Écrit Image2Ground=true dans le XML")
    ap.add_argument("--verbose", action="store_true",
                    help="Affiche les paramètres de rotation calculés")

    args = ap.parse_args()

    xyz_path = Path(args.xyz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = parse_xyz(xyz_path)
    if not rows:
        print("Aucune orientation trouvée.")
        sys.exit(2)

    effective_convention = args.convention
    if args.flip_image_180:
        if args.convention != "none":
            raise ValueError("--flip-image-180 et --convention ne doivent pas être utilisés ensemble")
        effective_convention = "flip_xy"

    for row in rows:
        image_name = row["label"]
        stem = Path(image_name).stem
        out_path = out_dir / f"{stem}.CON"

        M = build_rotation_matrix_photogrammetry(
            omega_deg=row["omega_deg"],
            phi_deg=row["phi_deg"],
            kappa_deg=row["kappa_deg"],
            euler_order=args.euler_order,
            kappa_offset_deg=args.kappa_offset,
            convention=effective_convention,
            convention_side=args.convention_side,
            transpose=args.transpose,
        )

        focal = args.focal if args.focal is not None else row["c"]

        if args.ppa_c is not None:
            ppa_c = args.ppa_c
        elif args.use_center_ppa:
            ppa_c = args.width / 2.0
        else:
            ppa_c = row["x0"]

        if args.ppa_l is not None:
            ppa_l = args.ppa_l
        elif args.use_center_ppa:
            ppa_l = args.height / 2.0
        else:
            ppa_l = -row["y0"]

        if args.verbose:
            print(
                f"{stem}: "
                f"omega={row['omega_deg']:.6f} "
                f"phi={row['phi_deg']:.6f} "
                f"kappa={row['kappa_deg']:.6f} "
                f"kappa_offset={args.kappa_offset:.6f} "
                f"ppa_c={ppa_c:.6f} "
                f"ppa_l={ppa_l:.6f} "
                f"convention={effective_convention}"
            )
            print(M)

        write_con_file(
            out_path=out_path,
            image_name=image_name,
            X0=row["X0"],
            Y0=row["Y0"],
            Z0=row["Z0"],
            M=M,
            width=args.width,
            height=args.height,
            focal=focal,
            ppa_c=ppa_c,
            ppa_l=ppa_l,
            geodesique=args.geodesique,
            euclidien_type=args.euclidien_type,
            sensor_name=args.sensor_name,
            pixel_size=args.pixel_size,
            image2ground=args.image2ground,
        )

    print(f"{len(rows)} fichier(s) .CON généré(s) dans {out_dir}")


if __name__ == "__main__":
    main()

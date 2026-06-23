#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python xyz2con.py \
  --input /Users/nbellaiche/DATA/dev/gsplat/dataset/drone/Image_orientations_dataset1.xyz \
  --images_dir /Users/nbellaiche/DATA/dev/gsplat/dataset/drone/Metashape_outputs_images \
  --output_dir /Users/nbellaiche/DATA/dev/gsplat/dataset/drone/Metashape_outputs_images \
  --order RxRyRz \
  --rot_flip_x \
  --flip_ppx

import argparse
import math
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image, ExifTags

def d2r(v): return v * math.pi / 180.0

def Rx(a):
    ca, sa = math.cos(a), math.sin(a)
    return [[1,0,0],[0,ca,-sa],[0,sa,ca]]

def Ry(a):
    ca, sa = math.cos(a), math.sin(a)
    return [[ca,0,sa],[0,1,0],[-sa,0,ca]]

def Rz(a):
    ca, sa = math.cos(a), math.sin(a)
    return [[ca,-sa,0],[sa,ca,0],[0,0,1]]

def mm(A,B):
    return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def opk_to_R(omega_deg, phi_deg, kappa_deg, order="RzRyRx"):
    o,p,k = d2r(omega_deg), d2r(phi_deg), d2r(kappa_deg)
    if order == "RzRyRx":
        return mm(mm(Rz(k), Ry(p)), Rx(o))
    elif order == "RxRyRz":
        return mm(mm(Rx(o), Ry(p)), Rz(k))
    raise ValueError("order must be RzRyRx or RxRyRz")

def diag3(sx, sy, sz):
    return [[sx,0,0],[0,sy,0],[0,0,sz]]

def indent(elem, level=0):
    i = "\n" + level*"    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        for e in elem:
            indent(e, level+1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

def fmt(v, n=12):
    return f"{float(v):.{n}f}"

def exif_dict(img):
    out = {}
    ex = img.getexif()
    if not ex:
        return out
    tagmap = {v:k for k,v in ExifTags.TAGS.items()}
    for name in ["FocalLengthIn35mmFilm", "FocalLength"]:
        tid = tagmap.get(name)
        if tid in ex:
            out[name] = ex.get(tid)
    return out

def exif_35mm_to_float(v):
    if v is None:
        return None
    try:
        if isinstance(v, tuple) and len(v) == 2:
            return float(v[0]) / float(v[1])
        return float(v)
    except Exception:
        return None

def estimate_focal_px_from_exif(width_px, exif35):
    if exif35 is None or exif35 <= 0:
        return None
    return (exif35 / 36.0) * float(width_px)

def build_con(row, width, height, c, l, focale, geodesique, order, rot_flip_x=False, rot_flip_y=False):
    R = opk_to_R(row["omega[deg]"], row["phi[deg]"], row["kappa[deg]"], order=order)

    # Flip axes caméra dans la rotation (hypothèses de convention)
    sx = -1 if rot_flip_x else 1
    sy = -1 if rot_flip_y else 1
    F = diag3(sx, sy, 1)
    R = mm(R, F)

    root = ET.Element("orientation")
    ET.SubElement(root, "lastmodificationbylibori", {"date":"2026-06-22","time":"00 h 00 min 00 sec"})
    ET.SubElement(root, "version").text = "1.0"

    aux = ET.SubElement(root, "auxiliarydata")
    img_name = Path(str(row["label"])).stem
    ET.SubElement(aux, "image_name").text = img_name
    idate = ET.SubElement(aux, "image_date")
    for t in ["year","month","day","hour","minute","second"]:
        ET.SubElement(idate, t).text = "0"
    ET.SubElement(idate, "time_system")
    ET.SubElement(aux, "samples")

    geom = ET.SubElement(root, "geometry", {"type":"physique"})
    ext = ET.SubElement(geom, "extrinseque")
    systeme = ET.SubElement(ext, "systeme")
    euc = ET.SubElement(systeme, "euclidien", {"type":"MATISRTL"})
    ET.SubElement(euc, "x").text = fmt(row["X0"], 12)
    ET.SubElement(euc, "y").text = fmt(row["Y0"], 12)
    ET.SubElement(systeme, "geodesique").text = geodesique
    ET.SubElement(ext, "grid_alti").text = "UNKNOWN"

    sommet = ET.SubElement(ext, "sommet")
    ET.SubElement(sommet, "easting").text = "0"
    ET.SubElement(sommet, "northing").text = "0"
    ET.SubElement(sommet, "altitude").text = fmt(row["Z0"], 12)

    rot = ET.SubElement(ext, "rotation")
    ET.SubElement(rot, "Image2Ground").text = "true"
    mat3d = ET.SubElement(rot, "mat3d")
    for i in range(3):
        li = ET.SubElement(mat3d, f"l{i+1}")
        pt = ET.SubElement(li, "pt3d")
        ET.SubElement(pt, "x").text = fmt(R[i][0], 15)
        ET.SubElement(pt, "y").text = fmt(R[i][1], 15)
        ET.SubElement(pt, "z").text = fmt(R[i][2], 15)

    intr = ET.SubElement(geom, "intrinseque")
    sensor = ET.SubElement(intr, "sensor")
    ET.SubElement(sensor, "name").text = "DRONE_CAMERA"

    calib = ET.SubElement(sensor, "calibration_date")
    for t in ["year","month","day","hour","minute","second"]:
        ET.SubElement(calib, t).text = "0"
    ET.SubElement(calib, "time_system")
    ET.SubElement(sensor, "serial_number").text = "UNKNOWN"

    imsz = ET.SubElement(sensor, "image_size")
    ET.SubElement(imsz, "width").text = str(int(width))
    ET.SubElement(imsz, "height").text = str(int(height))

    ssz = ET.SubElement(sensor, "sensor_size")
    ET.SubElement(ssz, "width").text = str(int(width))
    ET.SubElement(ssz, "height").text = str(int(height))

    ppa = ET.SubElement(sensor, "ppa")
    ET.SubElement(ppa, "c").text = fmt(c, 12)
    ET.SubElement(ppa, "l").text = fmt(l, 12)
    ET.SubElement(ppa, "focale").text = fmt(focale, 12)

    return ET.ElementTree(root), img_name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--order", default="RzRyRx", choices=["RzRyRx","RxRyRz"])
    ap.add_argument("--label", default=None)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--geodesique", default="LAMBERT93")
    ap.add_argument("--force_focale", type=float, default=None)

    # Hypothèses
    ap.add_argument("--flip_ppx", action="store_true", help="c <- width - c")
    ap.add_argument("--flip_ppy", action="store_true", help="l <- height - l")
    ap.add_argument("--rot_flip_x", action="store_true", help="flip axe x caméra dans R")
    ap.add_argument("--rot_flip_y", action="store_true", help="flip axe y caméra dans R")
    ap.add_argument("--kappa_sign", type=int, choices=[1,-1], default=1)
    ap.add_argument("--omega_sign", type=int, choices=[1,-1], default=1)
    ap.add_argument("--phi_sign", type=int, choices=[1,-1], default=1)

    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir)

    df = pd.read_csv(args.input, sep=r"\s+", engine="python")
    if "#label" in df.columns:
        df = df.rename(columns={"#label":"label"})

    req = ["label","X0","Y0","Z0","omega[deg]","phi[deg]","kappa[deg]"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise RuntimeError(f"Colonnes manquantes: {miss}")

    if args.label:
        target = Path(args.label).stem
        df = df[df["label"].apply(lambda v: Path(str(v)).stem == target)]
        if len(df) == 0:
            raise RuntimeError(f"Image {args.label} introuvable")

    n = 0
    for _, row in df.iterrows():
        row = row.copy()
        row["omega[deg]"] *= args.omega_sign
        row["phi[deg]"]   *= args.phi_sign
        row["kappa[deg]"] *= args.kappa_sign

        stem = Path(str(row["label"])).stem
        jpg = images_dir / f"{stem}.jpg"
        if not jpg.exists():
            raise RuntimeError(f"JPG introuvable: {jpg}")

        with Image.open(jpg) as im:
            width, height = im.size
            ex = exif_dict(im)

        c = width / 2.0
        l = height / 2.0
        if args.flip_ppx: c = width - c
        if args.flip_ppy: l = height - l

        if args.force_focale is not None:
            focale = float(args.force_focale)
        else:
            exif35 = exif_35mm_to_float(ex.get("FocalLengthIn35mmFilm"))
            focale = estimate_focal_px_from_exif(width, exif35) or 4636.912

        tree, img_name = build_con(
            row=row, width=width, height=height, c=c, l=l,
            focale=focale, geodesique=args.geodesique, order=args.order,
            rot_flip_x=args.rot_flip_x, rot_flip_y=args.rot_flip_y
        )
        indent(tree.getroot())
        tree.write(out / f"{img_name}.CON", encoding="utf-8", xml_declaration=True)
        n += 1

    print(f"OK: {n} fichier(s) écrit(s) dans {out}")

if __name__ == "__main__":
    main()

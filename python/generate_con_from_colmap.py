#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np

from con_camera_lib import (
    load_colmap_images,
    load_colmap_cameras,
    camera_to_intrinsics,
    qvec_to_rotmat,
    fit_conic_equivalent,
    validate_conic_equivalent,
    estimate_pixel_size_from_exif_and_colmap_focal,
    build_orientation_xml,
    prettify_xml,
)


def choose_reference_image(images, reference_name=None):
    if reference_name is not None:
        for _, im in images.items():
            if Path(im["name"]).name == reference_name or Path(im["name"]).stem == reference_name:
                return im
        raise ValueError(f"Image de référence introuvable: {reference_name}")

    # défaut: première image triée par nom
    ordered = sorted(images.values(), key=lambda d: d["name"])
    if not ordered:
        raise ValueError("Aucune image COLMAP trouvée")
    return ordered[0]


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Génère des .CON depuis COLMAP en ajustant une seule fois des intrinsics .CON "
            "équivalents sur une image de référence, puis en les réutilisant pour toutes les images."
        )
    )
    ap.add_argument("--colmap-dir", required=True, help="Dossier contenant sparse/0")
    ap.add_argument("--images-dir", required=True, help="Dossier des images source")
    ap.add_argument("--out", required=True, help="Dossier de sortie des .CON")
    ap.add_argument("--geodesic", default="LAMBERT93", help="Valeur du champ <geodesique>")
    ap.add_argument(
        "--reference-image",
        default=None,
        help="Nom ou stem de l'image de référence pour fitter les intrinsics .CON partagés",
    )
    ap.add_argument(
        "--grid-step",
        type=int,
        default=100,
        help="Pas de la grille d'échantillonnage en pixels pour le fit/validation",
    )
    ap.add_argument(
        "--focal-rel-span",
        type=float,
        default=0.08,
        help="Amplitude relative de recherche autour de la focale moyenne",
    )
    ap.add_argument(
        "--focal-steps",
        type=int,
        default=17,
        help="Nombre d'échantillons de focale testés",
    )
    ap.add_argument(
        "--pps-search-radius",
        type=float,
        default=80.0,
        help="Rayon de recherche PPS autour du PPA en pixels",
    )
    ap.add_argument(
        "--pps-step",
        type=float,
        default=10.0,
        help="Pas de recherche PPS en pixels",
    )
    args = ap.parse_args()

    colmap_dir = Path(args.colmap_dir)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = load_colmap_images(colmap_dir)
    cameras = load_colmap_cameras(colmap_dir)

    print(f"[INFO] images : {len(images)}")
    print(f"[INFO] cameras: {len(cameras)}")

    # -------------------------------------------------------------------------
    # Choix image/caméra de référence pour fitter UNE SEULE FOIS les intrinsics
    # -------------------------------------------------------------------------
    ref_im = choose_reference_image(images, args.reference_image)
    ref_cam = cameras[ref_im["camera_id"]]
    ref_intr = camera_to_intrinsics(ref_cam)

    print(
        f"[INFO] reference image: {ref_im['name']} "
        f"(camera_id={ref_im['camera_id']}, model={ref_cam['model']})"
    )

    shared_con_intr = fit_conic_equivalent(
        ref_intr,
        grid_step_px=args.grid_step,
        focal_rel_span=args.focal_rel_span,
        focal_steps=args.focal_steps,
        pps_search_radius_px=args.pps_search_radius,
        pps_step_px=args.pps_step,
    )
    shared_con_intr["width"] = ref_intr["width"]
    shared_con_intr["height"] = ref_intr["height"]

    validation = validate_conic_equivalent(
        shared_con_intr,
        ref_intr,
        step_px=args.grid_step,
    )

    print(
        "[INFO] shared intrinsics (.CON): "
        f"PPA=({shared_con_intr['ppa_c']:.3f},{shared_con_intr['ppa_l']:.3f}) "
        f"f={shared_con_intr['focal']:.3f} "
        f"PPS=({shared_con_intr['pps_c']:.3f},{shared_con_intr['pps_l']:.3f}) "
        f"r1={shared_con_intr['r1']:.3e} "
        f"r3={shared_con_intr['r3']:.3e}"
    )
    print(
        "[INFO] validation on reference: "
        f"rmse={validation['rmse']:.3f}px "
        f"mean={validation['mean']:.3f}px "
        f"max={validation['max']:.3f}px "
        f"n={validation['count']}"
    )

    # -------------------------------------------------------------------------
    # Génération de tous les .CON avec CES intrinsics partagés
    # -------------------------------------------------------------------------
    for _, im in sorted(images.items(), key=lambda kv: kv[1]["name"]):
        cam = cameras[im["camera_id"]]
        intr = camera_to_intrinsics(cam)

        # Vérification cohérence dimensionnelle
        if intr["width"] != shared_con_intr["width"] or intr["height"] != shared_con_intr["height"]:
            raise ValueError(
                "Toutes les images doivent partager la même taille pour réutiliser "
                "les mêmes intrinsics .CON. "
                f"Référence: {shared_con_intr['width']}x{shared_con_intr['height']}, "
                f"image {im['name']}: {intr['width']}x{intr['height']}"
            )

        R_cw = qvec_to_rotmat(im["qvec"])
        t = np.asarray(im["tvec"], dtype=np.float64)
        center = -R_cw.T @ t

        image_name_full = Path(im["name"]).name
        image_stem = Path(im["name"]).stem
        image_path = images_dir / image_name_full

        pixel_size_m = estimate_pixel_size_from_exif_and_colmap_focal(
            image_path=image_path,
            fx=intr["fx"],
            fy=intr["fy"],
        )

        if pixel_size_m is None:
            pixel_size_value = "UNKNOWN"
        else:
            pixel_size_value = f"{pixel_size_m:.15e}"

        root = build_orientation_xml(
            image_name=image_stem,
            center=center,
            R_cw=R_cw,
            con_intr=shared_con_intr,
            geodesic_name=args.geodesic,
            pixel_size_value=pixel_size_value,
        )

        xml_bytes = prettify_xml(root)
        out_path = out_dir / f"{image_stem}.CON"
        with open(out_path, "wb") as f:
            f.write(xml_bytes)

        print(
            f"[INFO] wrote {out_path.name}: "
            f"center=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f})"
        )

    print(f"[INFO] Fichiers .CON écrits dans {out_dir}")


if __name__ == "__main__":
    main()

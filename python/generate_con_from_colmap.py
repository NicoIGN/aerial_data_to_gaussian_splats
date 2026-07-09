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
    """
    - Si reference_name est fourni: match sur nom de fichier ou stem
    - Sinon: prend la première image dans l'ordre de lecture COLMAP
    """
    image_values = list(images.values())

    if reference_name is not None:
        for im in image_values:
            if Path(im["name"]).name == reference_name or Path(im["name"]).stem == reference_name:
                return im
        raise ValueError(f"Image de référence introuvable: {reference_name}")

    if not image_values:
        raise ValueError("Aucune image COLMAP trouvée")

    return image_values[0]


def build_recursive_image_index(image_dir: Path):
    """
    Construit deux index récursifs:
      - par nom de fichier
      - par stem
    Si collisions, on garde une liste de chemins.
    """
    by_name = {}
    by_stem = {}

    for p in image_dir.rglob("*"):
        if not p.is_file():
            continue

        name = p.name
        stem = p.stem

        by_name.setdefault(name, []).append(p)
        by_stem.setdefault(stem, []).append(p)

    return by_name, by_stem


def resolve_image_path(image_name_from_colmap: str, image_dir: Path, by_name, by_stem):
    """
    Cherche une image COLMAP:
      1. chemin direct sous image_dir
      2. récursivement par nom exact
      3. récursivement par stem
    """
    candidate_direct = image_dir / Path(image_name_from_colmap)
    if candidate_direct.exists() and candidate_direct.is_file():
        return candidate_direct

    base_name = Path(image_name_from_colmap).name
    stem = Path(image_name_from_colmap).stem

    if base_name in by_name:
        matches = by_name[base_name]
        if len(matches) == 1:
            return matches[0]
        return sorted(matches)[0]

    if stem in by_stem:
        matches = by_stem[stem]
        if len(matches) == 1:
            return matches[0]
        return sorted(matches)[0]

    return None


def compute_output_path(image_path: Path, image_stem: str, out_dir: Path | None):
    if out_dir is None:
        return image_path.parent / f"{image_stem}.CON"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{image_stem}.CON"


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Génère des .CON depuis COLMAP en ajustant une seule fois des intrinsics .CON "
            "équivalents sur une image de référence, puis en les réutilisant pour toutes les images."
        )
    )
    ap.add_argument("--colmap-dir", required=True, help="Dossier contenant sparse/0")
    ap.add_argument("--image-dir", required=True, help="Dossier racine des images source (scan récursif)")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Dossier de sortie des .CON. Si absent, écrit chaque .CON à côté de l'image source.",
    )
    ap.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help=(
            "Taille pixel en mètres. Si fournie, elle est utilisée telle quelle. "
            "Sinon, tentative d'estimation via la focale EXIF."
        ),
    )
    ap.add_argument("--geodesic", default="LAMBERT93", help="Valeur du champ <geodesique>")
    ap.add_argument(
        "--reference-image",
        default=None,
        help="Nom ou stem de l'image de référence. Par défaut: première image référencée dans COLMAP.",
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

    if args.pixel_size is not None and args.pixel_size <= 0:
        raise ValueError("--pixel-size doit être strictement positif")

    colmap_dir = Path(args.colmap_dir)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else None

    images = load_colmap_images(colmap_dir)
    cameras = load_colmap_cameras(colmap_dir)

    print(f"[INFO] images COLMAP : {len(images)}")
    print(f"[INFO] cameras       : {len(cameras)}")

    print(f"[INFO] scan récursif des images sous: {image_dir}")
    by_name, by_stem = build_recursive_image_index(image_dir)
    print(f"[INFO] index fichiers : {sum(len(v) for v in by_name.values())} fichiers trouvés")

    ref_im = choose_reference_image(images, args.reference_image)
    ref_cam = cameras[ref_im["camera_id"]]
    ref_intr = camera_to_intrinsics(ref_cam)

    print(
        f"[INFO] reference image COLMAP: {ref_im['name']} "
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

    if args.pixel_size is not None:
        forced_pixel_size_value = f"{float(args.pixel_size):.15e}"
        print(f"[INFO] pixel_size forcé par paramètre: {forced_pixel_size_value}")
    else:
        forced_pixel_size_value = None

    for _, im in images.items():
        cam = cameras[im["camera_id"]]
        intr = camera_to_intrinsics(cam)

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

        image_stem = Path(im["name"]).stem

        image_path = resolve_image_path(im["name"], image_dir, by_name, by_stem)
        if image_path is None:
            raise FileNotFoundError(
                f"Image source introuvable récursivement sous --image-dir pour {im['name']}"
            )

        if forced_pixel_size_value is not None:
            pixel_size_value = forced_pixel_size_value
        else:
            pixel_size_m = estimate_pixel_size_from_exif_and_colmap_focal(
                image_path=image_path,
                fx=intr["fx"],
                fy=intr["fy"],
            )
            if pixel_size_m is None:
                raise RuntimeError(
                    "Impossible d'estimer pixel_size à partir de l'EXIF pour "
                    f"l'image {image_path}. "
                    "Passe explicitement --pixel-size <valeur_en_metres>."
                )
            pixel_size_value = f"{pixel_size_m:.15e}"

        out_path = compute_output_path(image_path, image_stem, out_dir)

        root = build_orientation_xml(
            image_name=image_stem,
            center=center,
            R_cw=R_cw,
            con_intr=shared_con_intr,
            geodesic_name=args.geodesic,
            pixel_size_value=pixel_size_value,
        )

        xml_bytes = prettify_xml(root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(xml_bytes)

        print(
            f"[INFO] wrote {out_path}: "
            f"center=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f}) "
            f"image_path={image_path} "
            f"pixel_size={pixel_size_value}"
        )

    if out_dir is None:
        print("[INFO] Fichiers .CON écrits à côté des images source")
    else:
        print(f"[INFO] Fichiers .CON écrits dans {out_dir}")


if __name__ == "__main__":
    main()

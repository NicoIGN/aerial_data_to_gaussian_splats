#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

import warnings
from rasterio.errors import NotGeoreferencedWarning

try:
    import numpy as np
except ImportError:
    print("Erreur: numpy requis. Installe par exemple: pip install numpy")
    sys.exit(1)

try:
    import rasterio
    from rasterio.enums import Resampling
except ImportError:
    print("Erreur: rasterio requis. Installe par exemple:")
    print("  conda install -c conda-forge rasterio")
    print("ou")
    print("  pip install rasterio")
    sys.exit(1)

try:
    from PIL import Image, ImageFile
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError:
    print("Erreur: pillow requis. Installe par exemple: pip install pillow")
    sys.exit(1)


SUPPORTED_EXTENSIONS = {
    ".jp2", ".j2k", ".jpf", ".jpx",
    ".tif", ".tiff",
    ".png",
    ".jpg", ".jpeg",
    ".bmp",
    ".webp",
}

JP2_EXTENSIONS = {".jp2", ".j2k", ".jpf", ".jpx"}


def verbose_print(message: str, verbose: bool):
    """Affiche un message uniquement si le mode verbose est actif."""
    if verbose:
        print(message)


def ensure_driver_support(image_path: Path, verbose: bool = False):
    """Vérifie si Rasterio supporte les fichiers JP2."""
    ext = image_path.suffix.lower()
    if ext not in JP2_EXTENSIONS:  # Si ce n'est pas du JP2, ok sans vérification
        return

    try:
        with rasterio.open(image_path) as _:
            verbose_print(f"[VERBOSE] Le fichier {image_path} peut être ouvert avec Rasterio.", verbose)
    except rasterio.errors.RasterioIOError:
        raise RuntimeError(
            f"Erreur: le support JPEG2000 semble absent dans GDAL/rasterio.\n"
            f"Fichier problématique: {image_path}\n"
            "Assurez-vous que GDAL/rasterio supporte le driver JP2OpenJPEG (ou équivalent). "
            "Avec conda, essayez :\n"
            "  conda install -c conda-forge rasterio gdal libgdal"
        )


def normalize_to_uint8(arr: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Convertit une image en tableau numpy uint8, gérant les valeurs extrêmes."""
    if verbose:
        print("[VERBOSE] Conversion de l'image en uint8 pour l'affichage.")

    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)
    amin = np.nanmin(arr)
    amax = np.nanmax(arr)

    if verbose:
        print(f"[VERBOSE] Min pixel: {amin:.2f}, Max pixel: {amax:.2f}")

    if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = (255.0 * (arr - amin) / (amax - amin)).clip(0, 255)
    return arr.astype(np.uint8)


def read_preview_with_rasterio(image_path: Path, max_size: int, verbose: bool = False) -> Image.Image:
    """Lit une image avec Rasterio et génère un tableau réduit (preview)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)

        verbose_print(f"[VERBOSE] Lecture de l'image {image_path} avec Rasterio.", verbose)

        with rasterio.open(image_path) as ds:
            src_w, src_h = ds.width, ds.height

            verbose_print(f"[VERBOSE] Dimensions originales : {src_w}x{src_h}", verbose)

            scale = min(max_size / max(src_w, src_h), 1.0)
            out_w = max(1, int(round(src_w * scale)))
            out_h = max(1, int(round(src_h * scale)))

            verbose_print(f"[VERBOSE] Dimensions redimensionnées : {out_w}x{out_h}", verbose)

            count = ds.count

            if count >= 3:
                bands = [1, 2, 3]
                arr = ds.read(bands, out_shape=(3, out_h, out_w), resampling=Resampling.nearest)
                arr = np.transpose(arr, (1, 2, 0))
            elif count == 1:
                band = ds.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)
                arr = np.stack([band, band, band], axis=-1)
            else:
                raise ValueError(f"Nombre de bandes non supporté pour {image_path} : {count}")

            arr = normalize_to_uint8(arr, verbose=verbose)

            verbose_print("[VERBOSE] Lecture terminée, image prête pour le rendu.", verbose)
            return Image.fromarray(arr, mode="RGB")


def ensure_supported_input(image_path: Path, verbose: bool = False):
    """Vérifie que le fichier existe et possède une extension supportée."""
    verbose_print(f"[VERBOSE] Vérification de l'image : {image_path}", verbose)
    if not image_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {image_path}")

    ext = image_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Extension non supportée: {ext}\n"
            f"Extensions supportées : {supported}"
        )

    ensure_driver_support(image_path, verbose=verbose)


def infer_output_path(input_path: Path, output_dir: Path, use_png: bool) -> Path:
    """Génère le chemin de sortie."""
    ext = ".png" if use_png else ".jpg"
    return output_dir / f"{input_path.stem}{ext}"


def save_preview(img: Image.Image, output_path: Path, use_png: bool, verbose: bool = False):
    """Sauvegarde l'image générée dans le format demandé."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    verbose_print(f"[VERBOSE] Sauvegarde de l'image préview : {output_path}", verbose)

    if use_png:
        img.save(output_path)
    else:
        img.save(output_path, quality=90)


def main():
    """Point d'entrée principal."""
    ap = argparse.ArgumentParser(description="Génère un preview d'image avec Rasterio.")
    ap.add_argument("--input", required=True, help="Image source en entrée.")
    ap.add_argument("--output-dir", required=True, help="Dossier pour sauvegarder les previews.")
    ap.add_argument("--max-size", type=int, default=512, help="Taille max (plus grand côté).")
    ap.add_argument("--png", action="store_true", help="Génère un fichier PNG.")
    ap.add_argument("--jpg", action="store_true", help="Génère un fichier JPEG (par défaut).")
    ap.add_argument("--verbose", action="store_true", help="Affiche des messages détaillés.")

    args = ap.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    try:
        ensure_supported_input(input_path, verbose=args.verbose)
    except Exception as e:
        print(f"Erreur: {e}")
        sys.exit(1)

    use_png = args.png
    output_path = infer_output_path(input_path, output_dir, use_png)

    verbose_print(f"[1/2] Lecture preview : {input_path}", args.verbose)
    try:
        img = read_preview_with_rasterio(input_path, max_size=args.max_size, verbose=args.verbose)
    except Exception as e:
        print(f"Erreur pendant la lecture/génération du preview : {e}")
        sys.exit(1)

    verbose_print(f"[2/2] Écriture preview : {output_path}", args.verbose)
    try:
        save_preview(img, output_path, use_png, verbose=args.verbose)
    except Exception as e:
        print(f"Erreur pendant l'écriture du preview : {e}")
        sys.exit(1)

    # print("Terminé.")
    if args.verbose:
        print(f"[INFO] Preview généré : {output_path}")


if __name__ == "__main__":
    main()

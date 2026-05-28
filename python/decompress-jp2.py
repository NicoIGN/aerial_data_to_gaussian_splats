#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import warnings
from pathlib import Path
from PIL import Image
import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.errors import NotGeoreferencedWarning
except ImportError:
    print("Erreur: rasterio requis.")
    print("conda install -c conda-forge rasterio")
    sys.exit(1)

try:
    from PIL import Image, ImageFile
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError:
    print("Erreur: pillow requis.")
    print("pip install pillow")
    sys.exit(1)


JP2_EXTENSIONS = {".jp2", ".j2k", ".jpf", ".jpx"}

# suppression globale robuste
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def verbose_print(message: str, verbose: bool):
    if verbose:
        print(message)


def sanitize_cli_path(raw: str) -> Path:
    if raw is None:
        raise ValueError("Path argument is None")

    cleaned = raw.replace("^C", "").strip()

    if cleaned != raw:
        print(f"⚠️ Path corrigé automatiquement: '{raw}' -> '{cleaned}'")

    return Path(cleaned).expanduser().resolve()


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)

    amin = np.nanmin(arr)
    amax = np.nanmax(arr)

    if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = (255.0 * (arr - amin) / (amax - amin)).clip(0, 255)
    return arr.astype(np.uint8)


def is_valid_output(image_path: Path, expected_size: tuple[int, int], verbose: bool = False) -> bool:
    if not image_path.exists():
        return False

    try:
        with Image.open(image_path) as img:
            img.verify()

        with Image.open(image_path) as img:
            w, h = img.size

        if verbose:
            print(f"   🔍 existing: {w}x{h}, expected: {expected_size[0]}x{expected_size[1]}")

        return (w, h) == expected_size

    except Exception as e:
        if verbose:
            print(f"   ⚠️ invalid image: {e}")
        return False


def get_raster_size(image_path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NotGeoreferencedWarning)
        with rasterio.open(image_path) as ds:
            return ds.width, ds.height


def read_downsampled_image(
    image_path: Path,
    factor: float,
    verbose: bool = False,
) -> Image.Image:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NotGeoreferencedWarning)

        verbose_print(f"📖 Reading {image_path}", verbose)

        with rasterio.open(image_path) as ds:
            src_w = ds.width
            src_h = ds.height

            if factor <= 1:
                out_w = src_w
                out_h = src_h
            else:
                out_w = max(1, int(src_w / factor))
                out_h = max(1, int(src_h / factor))

            verbose_print(
                f"   {src_w}x{src_h} → {out_w}x{out_h}",
                verbose,
            )

            band_count = ds.count

            if band_count >= 3:
                arr = ds.read(
                    [1, 2, 3],
                    out_shape=(3, out_h, out_w),
                    resampling=Resampling.bilinear,
                )
                arr = np.transpose(arr, (1, 2, 0))

            elif band_count == 1:
                band = ds.read(
                    1,
                    out_shape=(out_h, out_w),
                    resampling=Resampling.bilinear,
                )
                arr = np.stack([band, band, band], axis=-1)

            else:
                raise RuntimeError(f"Unsupported band count: {band_count}")

            arr = normalize_to_uint8(arr)
            return Image.fromarray(arr, mode="RGB")


def save_image(
    img: Image.Image,
    output_path: Path,
    output_format: str,
    jpeg_quality: int,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "png":
        img.save(output_path)

    elif output_format == "jpg":
        img.save(
            output_path,
            quality=jpeg_quality,
            optimize=True,
        )

    elif output_format == "tiff":
        img.save(output_path)


def get_output_extension(fmt: str) -> str:
    mapping = {
        "png": ".png",
        "jpg": ".jpg",
        "tiff": ".tif",
    }
    return mapping[fmt]


def process_file(
    input_path: Path,
    input_dir: Path,
    output_dir: Path,
    factor: float,
    output_format: str,
    jpeg_quality: int,
    verbose: bool = False,
):
    rel_path = input_path.relative_to(input_dir)

    output_path = output_dir / rel_path.with_suffix(get_output_extension(output_format))

    try:
        src_w, src_h = get_raster_size(input_path)
    except Exception:
        src_w, src_h = None, None

    if src_w is not None:
        if factor <= 1:
            expected_w, expected_h = src_w, src_h
        else:
            expected_w = max(1, int(src_w / factor))
            expected_h = max(1, int(src_h / factor))
    else:
        expected_w, expected_h = None, None

    if output_path.exists() and expected_w is not None:
        if is_valid_output(output_path, (expected_w, expected_h), verbose=verbose):
            verbose_print(f"⏭️ Skip valid cache {output_path}", verbose)
            return
        else:
            verbose_print(f"♻️ Corrupt/invalid cache, recomputing {output_path}", verbose)

    img = read_downsampled_image(
        image_path=input_path,
        factor=factor,
        verbose=verbose,
    )

    save_image(
        img=img,
        output_path=output_path,
        output_format=output_format,
        jpeg_quality=jpeg_quality,
    )

    verbose_print(f"✅ Saved {output_path}", verbose)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decompress and downsample JP2 images."
    )

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--factor",
        type=float,
        default=1.0,
        help="Downsample factor (2 = divide W/H by 2)",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--png", action="store_true")
    group.add_argument("--jpg", action="store_true")
    group.add_argument("--tiff", action="store_true")

    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = sanitize_cli_path(args.input_dir)
    output_dir = sanitize_cli_path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    if args.png:
        output_format = "png"
    elif args.jpg:
        output_format = "jpg"
    else:
        output_format = "tiff"

    files = sorted(
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in JP2_EXTENSIONS
    )

    if not files:
        print(f"❌ No JP2 found in: {input_dir}")
        return

    print("================================")
    print("🖼️ JP2 DECOMPRESSION")
    print("================================")
    print(f"Input dir    : {input_dir}")
    print(f"Output dir   : {output_dir}")
    print(f"Factor       : {args.factor}")
    print(f"Output format: {output_format}")
    print(f"Files found  : {len(files)}")
    print("================================")

    failed = []

    for i, file in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {file.name}")

        try:
            process_file(
                input_path=file,
                input_dir=input_dir,
                output_dir=output_dir,
                factor=args.factor,
                output_format=output_format,
                jpeg_quality=args.jpeg_quality,
                verbose=args.verbose,
            )
        except Exception as e:
            failed.append((file, str(e)))
            print(f"❌ Failed: {file}")
            print(f"   {e}")

    print("\n================================")
    print("✅ DONE")
    print("================================")
    print(f"Success : {len(files) - len(failed)}")
    print(f"Failed  : {len(failed)}")

    if failed:
        print("\nFailed files:")
        for f, err in failed:
            print(f" - {f}")
            print(f"   {err}")


if __name__ == "__main__":
    main()

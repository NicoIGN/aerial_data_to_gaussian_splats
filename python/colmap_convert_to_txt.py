#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convertit cameras.bin, images.bin et points3D.bin de COLMAP en fichiers texte."
    )
    parser.add_argument(
        "colmap_dir",
        help="Répertoire COLMAP contenant sparse/0/"
    )
    parser.add_argument(
        "--colmap",
        default="colmap",
        help="Exécutable COLMAP (défaut : colmap)"
    )
    args = parser.parse_args()

    sparse_dir = Path(args.colmap_dir) / "sparse" / "0"

    if not sparse_dir.is_dir():
        print(f"Erreur : {sparse_dir} n'existe pas.", file=sys.stderr)
        sys.exit(1)

    required_files = [
        sparse_dir / "cameras.bin",
        sparse_dir / "images.bin",
        sparse_dir / "points3D.bin",
    ]

    missing = [f.name for f in required_files if not f.exists()]
    if missing:
        print(
            f"Erreur : fichiers manquants dans {sparse_dir} : {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = [
        args.colmap,
        "model_converter",
        "--input_path", str(sparse_dir),
        "--output_path", str(sparse_dir),
        "--output_type", "TXT",
    ]

    print("Exécution :", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

    print("\nConversion terminée.")
    print("Fichiers générés :")
    print(sparse_dir / "cameras.txt")
    print(sparse_dir / "images.txt")
    print(sparse_dir / "points3D.txt")


if __name__ == "__main__":
    main()

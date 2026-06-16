#!/usr/bin/env python3

import argparse
import sys
import numpy as np

from plyfile import PlyData, PlyElement


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convertit un fichier PLY ASCII/Binaire en conservant toutes les propriétés."
    )

    parser.add_argument(
        "input",
        help="PLY d'entrée"
    )

    parser.add_argument(
        "output",
        help="PLY de sortie"
    )

    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument(
        "--ascii",
        action="store_true",
        help="Écrire le fichier en ASCII"
    )

    mode.add_argument(
        "--bin",
        action="store_true",
        help="Écrire le fichier en binaire"
    )

    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        metavar="N",
        help="Conserver 1 vertex sur N (défaut: 1)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.subsample < 1:
        print("Erreur: --subsample doit être >= 1")
        sys.exit(1)

    print(f"Lecture : {args.input}")

    ply = PlyData.read(args.input)

    print(f"Format source : {'ASCII' if ply.text else 'BINAIRE'}")
    print(f"Nombre d'éléments : {len(ply.elements)}")

    new_elements = []

    for element in ply.elements:

        name = element.name
        data = element.data

        print(f"  - {name}: {len(data):,} entrées")

        if name == "vertex" and args.subsample > 1:

            original_count = len(data)

            data = data[::args.subsample].copy()

            print(
                f"    Sous-échantillonnage 1/{args.subsample}: "
                f"{original_count:,} -> {len(data):,}"
            )

        new_elements.append(
            PlyElement.describe(
                data,
                name,
                comments=element.comments
            )
        )

    output_text = args.ascii

    print(
        f"Écriture : {args.output} "
        f"({'ASCII' if output_text else 'BINAIRE'})"
    )

    out_ply = PlyData(
        new_elements,
        text=output_text,
        comments=ply.comments,
        obj_info=ply.obj_info
    )

    out_ply.write(args.output)

    print("Terminé.")


if __name__ == "__main__":
    main()

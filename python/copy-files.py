#!/usr/bin/env python3
import csv
import os
import shutil
import argparse

def main(input_csv, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    copied = 0
    missing = 0

    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        if "Cliche" not in reader.fieldnames:
            raise ValueError("Colonne 'Cliche' introuvable dans le CSV")

        for row in reader:
            cliche = row["Cliche"].strip()

            if not cliche:
                continue

            filename = f"{cliche}.jp2"
            src = os.path.join(input_dir, filename)
            dst = os.path.join(output_dir, filename)

            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied += 1
            else:
                missing += 1
                print(f"[WARN] fichier introuvable: {src}")

    print("\n--- Résumé ---")
    print(f"Fichiers copiés : {copied}")
    print(f"Manquants       : {missing}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copie des fichiers .jp2 à partir du champ Cliche d'un CSV")
    parser.add_argument("--csv", required=True, help="Chemin du fichier CSV")
    parser.add_argument("--input", required=True, help="Dossier contenant les fichiers source")
    parser.add_argument("--output", required=True, help="Dossier de destination")

    args = parser.parse_args()

    main(args.csv, args.input, args.output)

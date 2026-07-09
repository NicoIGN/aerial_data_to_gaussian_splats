#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import numpy as np
from pathlib import Path

from con_camera_lib import parse_con_orientation, con_project_world_to_image


def main():
    ap = argparse.ArgumentParser(
        description="Projette un point 3D avec le même modèle .CON que celui utilisé par le générateur."
    )
    ap.add_argument("--con", required=True)
    ap.add_argument("--x", type=float, required=True)
    ap.add_argument("--y", type=float, required=True)
    ap.add_argument("--z", type=float, required=True)
    ap.add_argument("--obs-c", type=float, default=None)
    ap.add_argument("--obs-l", type=float, default=None)
    args = ap.parse_args()

    con_path = Path(args.con)
    ori = parse_con_orientation(con_path)
    xyz = np.array([args.x, args.y, args.z], dtype=np.float64)

    uv, Xc = con_project_world_to_image(ori, xyz)

    print("===== ORIENTATION =====")
    print(f"file      : {con_path}")
    print(f"size      : {ori['width']} x {ori['height']}")
    print(f"PPA       : ({ori['ppa_c']:.6f}, {ori['ppa_l']:.6f})")
    print(f"PPS       : ({ori['pps_c']:.6f}, {ori['pps_l']:.6f})")
    print(f"focal     : {ori['focal']:.6f}")
    print(f"dist      : r1={ori['r1']:.12e}, r3={ori['r3']:.12e}, r5={ori['r5']:.12e}, r7={ori['r7']:.12e}")

    print()
    print("===== POINT 3D =====")
    print(f"xyz       : ({xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f})")

    print()
    print("===== PROJECTION =====")
    if uv is None:
        print("Projection invalide: point derrière la caméra.")
        return

    print(f"Xc        : ({Xc[0]:.6f}, {Xc[1]:.6f}, {Xc[2]:.6f})")
    print(f"image     : ({uv[0]:.6f}, {uv[1]:.6f})")
    inside = (0.0 <= uv[0] < ori["width"]) and (0.0 <= uv[1] < ori["height"])
    print(f"inside    : {inside}")

    if args.obs_c is not None and args.obs_l is not None:
        obs = np.array([args.obs_c, args.obs_l], dtype=np.float64)
        err = obs - uv
        norm = math.hypot(err[0], err[1])

        print()
        print("===== COMPARAISON =====")
        print(f"attendu   : ({obs[0]:.6f}, {obs[1]:.6f})")
        print(f"delta     : ({err[0]:.6f}, {err[1]:.6f})")
        print(f"erreur    : {norm:.6f} px")


if __name__ == "__main__":
    main()

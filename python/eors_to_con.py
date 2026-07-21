#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import re
import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

from con_camera_lib import (
    build_orientation_xml,
    prettify_xml,
)

# ---------------------------------------------------------------------------
IMAGE_EXTS = re.compile(r"\.(jpe?g|tiff?|png|bmp|jp2)$", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------

def _Rx(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], dtype=np.float64)

def _Ry(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]], dtype=np.float64)

def _Rz(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]], dtype=np.float64)

ROTATION_ORDERS = {
    "ZYX": lambda o,p,k: _Rz(k) @ _Ry(p) @ _Rx(o),
    "XYZ": lambda o,p,k: _Rx(o) @ _Ry(p) @ _Rz(k),
    "ZXY": lambda o,p,k: _Rz(k) @ _Rx(o) @ _Ry(p),
    "YXZ": lambda o,p,k: _Ry(p) @ _Rx(o) @ _Rz(k),
    "XZY": lambda o,p,k: _Rx(o) @ _Rz(k) @ _Ry(p),
    "YZX": lambda o,p,k: _Ry(p) @ _Rz(k) @ _Rx(o),
}

FLIPS = {
    f"S({''.join(n for n,v in zip('XYZ',[sx,sy,sz]) if v<0) or 'I'})": np.diag([sx,sy,sz])
    for sx,sy,sz in product([1,-1], repeat=3)
}

# Convention obs paramétrée par height (connu au moment de l'appel)
# obs(c, h-l) : passage repère direct → repère image indirect (.CON)
def make_obs_convs(height: int):
    h = float(height)
    return {
        "obs(c,h-l)": lambda c, l: (c, h - l),   # ← bonne convention physique
        "obs(c,l)":   lambda c, l: (c, l),
        "obs(c,-l)":  lambda c, l: (c, -l),
    }

def build_R_cw(omega_deg, phi_deg, kappa_deg, rot_fn, flip):
    o = math.radians(float(omega_deg))
    p = math.radians(float(phi_deg))
    k = math.radians(float(kappa_deg))
    return flip @ rot_fn(o, p, k).T


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_eors(path: Path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
    if not first.startswith("#"):
        raise RuntimeError(f"Header '#' attendu, trouvé: {first}")
    header_cols = re.split(r"\s+", first[1:].strip())
    df = pd.read_csv(path, sep=r"\s+", engine="python",
                     comment="#", header=None, names=header_cols)
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "pointlabel":       rename[c] = "label"
        elif cl == "x0":             rename[c] = "X0"
        elif cl == "y0":             rename[c] = "Y0"
        elif cl == "z0":             rename[c] = "Z0"
        elif cl.startswith("omega"): rename[c] = "omega_deg"
        elif cl.startswith("phi"):   rename[c] = "phi_deg"
        elif cl.startswith("kappa"): rename[c] = "kappa_deg"
    df = df.rename(columns=rename)
    required = ["label", "X0", "Y0", "Z0", "omega_deg", "phi_deg", "kappa_deg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Colonnes manquantes: {missing}")
    return df


def load_tp3d(path: Path):
    pts = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.replace("\t", " ").split()
            if len(p) < 4:
                continue
            try:
                pts[int(float(p[0]))] = (float(p[1]), float(p[2]), float(p[3]))
            except Exception:
                pass
    return pts


def build_file_index(images_dir: Path):
    candidates: dict[str, list[Path]] = {}
    by_obs: dict[str, Path] = {}
    for p in images_dir.rglob("*"):
        if not p.is_file():
            continue
        if IMAGE_EXTS.search(p.suffix):
            candidates.setdefault(p.stem, []).append(p)
        elif p.suffix.lower() == ".txt" and p.stem.endswith("_obs"):
            by_obs[p.stem[:-len("_obs")]] = p
    by_image = {stem: sorted(paths)[0] for stem, paths in candidates.items()}
    return by_image, by_obs


def choose_reference_row(eors_df, by_image, reference_label=None):
    if reference_label is not None:
        for row in eors_df.itertuples(index=False):
            stem = Path(str(row.label)).stem
            if (str(row.label) == reference_label or stem == reference_label) \
                    and stem in by_image:
                return row
        raise ValueError(f"Référence '{reference_label}' introuvable.")
    for row in eors_df.itertuples(index=False):
        if Path(str(row.label)).stem in by_image:
            return row
    raise ValueError("Aucune image du EORS trouvée dans --images.")


def get_image_size(image_path: Path):
    from PIL import Image
    with Image.open(image_path) as img:
        return img.size


def parse_obs_file(obs_path: Path, max_pts: int = 0):
    rows = []
    with open(obs_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.replace("\t", " ").split()
            if len(p) < 3:
                continue
            try:
                rows.append((int(float(p[0])), float(p[1]), float(p[2])))
            except Exception:
                pass
    if max_pts > 0 and len(rows) > max_pts:
        idx = np.linspace(0, len(rows)-1, max_pts, dtype=int)
        rows = [rows[i] for i in idx]
    return rows


# ---------------------------------------------------------------------------
# Dataset de calibration
# ---------------------------------------------------------------------------

def build_calib_dataset(eors_df, tp3d, by_obs,
                         rot_fn, flip, obs_fn,
                         max_pts_per_image=0):
    rows = []
    for row in eors_df.itertuples(index=False):
        stem     = Path(str(row.label)).stem
        obs_path = by_obs.get(stem)
        if obs_path is None:
            continue
        obs_rows = parse_obs_file(obs_path, max_pts=max_pts_per_image)
        if not obs_rows:
            continue

        center = np.array([float(row.X0), float(row.Y0), float(row.Z0)],
                           dtype=np.float64)
        R_cw   = build_R_cw(row.omega_deg, row.phi_deg, row.kappa_deg, rot_fn, flip)
        tvec   = -R_cw @ center

        for pid, c_obs, l_obs in obs_rows:
            xyz = tp3d.get(pid)
            if xyz is None:
                continue
            Xc = R_cw @ np.asarray(xyz, dtype=np.float64) + tvec
            zb = float(Xc[2])
            if zb <= 1e-9:
                continue
            u_obs, v_obs = obs_fn(float(c_obs), float(l_obs))
            rows.append([float(Xc[0])/zb, float(Xc[1])/zb, u_obs, v_obs])

    if not rows:
        return None
    return np.asarray(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Fit pinhole linéaire
# ---------------------------------------------------------------------------

def fit_pinhole(data):
    xn, yn, u_obs, v_obs = data[:,0], data[:,1], data[:,2], data[:,3]
    N = len(xn)
    A = np.zeros((2*N, 3), dtype=np.float64)
    b = np.zeros(2*N,      dtype=np.float64)
    A[:N,0]=1.0;  A[:N,2]=xn;  b[:N]=u_obs
    A[N:,1]=1.0;  A[N:,2]=yn;  b[N:]=v_obs
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    ppa_c, ppa_l, f = x
    res_u = u_obs - (ppa_c + f*xn)
    res_v = v_obs - (ppa_l + f*yn)
    err   = np.sqrt(res_u**2 + res_v**2)
    return float(f), float(ppa_c), float(ppa_l), float(np.sqrt(np.mean(err**2)))


# ---------------------------------------------------------------------------
# Recherche convention
# ---------------------------------------------------------------------------

def search_best_convention(eors_df, tp3d, by_obs, height,
                            search_max_pts=50, rmse_threshold=100.0):
    obs_convs = make_obs_convs(height)
    results = []
    total = len(ROTATION_ORDERS) * len(FLIPS) * len(obs_convs)
    done  = 0

    for rot_name, rot_fn in ROTATION_ORDERS.items():
        for flip_name, flip_mat in FLIPS.items():
            for obs_name, obs_fn in obs_convs.items():
                data = build_calib_dataset(
                    eors_df, tp3d, by_obs,
                    rot_fn, flip_mat, obs_fn,
                    max_pts_per_image=search_max_pts,
                )
                done += 1
                rmse = float("inf")
                if data is not None and len(data) >= 10:
                    f_val, _, _, rmse = fit_pinhole(data)
                    if f_val < 0:
                        rmse = float("inf")  # focale négative → invalide
                results.append((rmse, rot_name, flip_name, obs_name))
                print(f"\r  {done}/{total}  best: "
                      f"{min(results)[1]}+{min(results)[2]}+{min(results)[3]} "
                      f"rmse={min(results)[0]:.2f}px    ",
                      end="", flush=True)

    print()
    results.sort()

    print("\n" + "=" * 65)
    print("CLASSEMENT PINHOLE (top 10, sans distorsion, f > 0 uniquement)")
    print("=" * 65)
    print(f"{'#':>3}  {'rotation':>10}  {'flip':>8}  {'obs':>12}  {'rmse':>8}")
    print("-" * 65)
    for i, (rmse, rot, flip, obs) in enumerate(results[:10], 1):
        mark = "" if rmse < float("inf") else "  (f<0)"
        print(f"{i:>3}  {rot:>10}  {flip:>8}  {obs:>12}  {rmse:>8.3f}px{mark}")
    print("=" * 65)

    best_rmse, best_rot, best_flip, best_obs = results[0]
    print(f"\n[INFO] Meilleure convention : "
          f"rotation={best_rot}  flip={best_flip}  obs={best_obs}  "
          f"rmse={best_rmse:.3f}px")

    if best_rmse > rmse_threshold:
        raise RuntimeError(
            f"RMSE pinhole = {best_rmse:.2f}px > seuil {rmse_threshold}px.\n"
            f"Aucune convention valide (f>0) trouvée sous le seuil.\n"
            f"Vérifiez le système de coordonnées ou la qualité du tp3d."
        )

    obs_convs_full = make_obs_convs(height)
    return (ROTATION_ORDERS[best_rot], FLIPS[best_flip],
            obs_convs_full[best_obs],
            best_rot, best_flip, best_obs, best_rmse)


# ---------------------------------------------------------------------------
# Fit PPS
# ---------------------------------------------------------------------------

def fit_pps_grid(data, f, ppa_c, ppa_l, search_radius=300.0, step=20.0):
    xn, yn, u_obs, v_obs = data[:,0], data[:,1], data[:,2], data[:,3]
    u0 = ppa_c + f * xn;  v0 = ppa_l + f * yn
    offsets = np.arange(-search_radius, search_radius + step*0.5, step)
    best_score = float("inf")
    best_dc, best_dl = 0.0, 0.0
    for dc in offsets:
        for dl in offsets:
            pps_c = ppa_c + dc;  pps_l = ppa_l + dl
            du = u0 - pps_c;  dv = v0 - pps_l
            R  = np.sqrt(du**2 + dv**2)
            valid = R > 1.0
            if not np.any(valid):
                continue
            ru = (u_obs - u0)[valid]; rv = (v_obs - v0)[valid]
            tang = (-dv[valid]*ru + du[valid]*rv) / R[valid]
            score = float(np.mean(tang**2))
            if score < best_score:
                best_score = score
                best_dc, best_dl = dc, dl
    return ppa_c + best_dc, ppa_l + best_dl, best_score


def refine_pps(data, f, ppa_c, ppa_l, pps_c0, pps_l0, num_iter=20):
    xn, yn, u_obs, v_obs = data[:,0], data[:,1], data[:,2], data[:,3]
    u0 = ppa_c + f * xn;  v0 = ppa_l + f * yn
    pps_c, pps_l = pps_c0, pps_l0
    for _ in range(num_iter):
        du = u0 - pps_c;  dv = v0 - pps_l
        R  = np.sqrt(du**2 + dv**2)
        valid = R > 1.0
        ru = (u_obs - u0)[valid]; rv = (v_obs - v0)[valid]
        tang = (-dv[valid]*ru + du[valid]*rv) / R[valid]
        duv = du[valid]; dvv = dv[valid]; Rv = R[valid]
        J_pc = ( dvv*ru - duv*rv) / Rv
        J_pl = (-dvv*rv - duv*ru) / Rv
        JtJ  = np.array([[np.sum(J_pc**2),    np.sum(J_pc*J_pl)],
                          [np.sum(J_pc*J_pl),  np.sum(J_pl**2)  ]])
        Jtr  = np.array([np.sum(J_pc*tang), np.sum(J_pl*tang)])
        try:
            delta = np.linalg.solve(JtJ + 1e-6*np.eye(2), Jtr)
        except np.linalg.LinAlgError:
            break
        pps_c -= delta[0];  pps_l -= delta[1]
    return float(pps_c), float(pps_l)


# ---------------------------------------------------------------------------
# Fit distorsion radiale
# ---------------------------------------------------------------------------

def fit_distortion(data, f, ppa_c, ppa_l, pps_c, pps_l):
    xn, yn, u_obs, v_obs = data[:,0], data[:,1], data[:,2], data[:,3]
    u0 = ppa_c + f*xn;  v0 = ppa_l + f*yn
    du = u0 - pps_c;  dv = v0 - pps_l
    R2 = (du**2 + dv**2) / (f*f)
    rhs_u = (u_obs - pps_c) - du
    rhs_v = (v_obs - pps_l) - dv
    N = len(xn)
    A = np.zeros((2*N, 4), dtype=np.float64)
    rhs = np.zeros(2*N, dtype=np.float64)
    for j, power in enumerate([1, 2, 3, 4]):
        Rp = R2**power
        A[:N, j] = du*Rp;  A[N:, j] = dv*Rp
    rhs[:N] = rhs_u;  rhs[N:] = rhs_v
    lam = 1e-4
    coeffs, *_ = np.linalg.lstsq(A.T@A + lam*np.eye(4), A.T@rhs, rcond=None)
    return coeffs


# ---------------------------------------------------------------------------
# Projection .CON
# ---------------------------------------------------------------------------

def _con_distort(u0, v0, con_intr):
    pps_c = con_intr["pps_c"];  pps_l = con_intr["pps_l"]
    r1    = con_intr["r1"];     r3    = con_intr["r3"]
    r5    = con_intr.get("r5", 0.0);  r7 = con_intr.get("r7", 0.0)
    f     = con_intr["focal"]
    du, dv = u0 - pps_c, v0 - pps_l
    R2 = (du**2 + dv**2) / (f*f)
    sc = 1.0 + r1*R2 + r3*R2**2 + r5*R2**3 + r7*R2**4
    return pps_c + du*sc, pps_l + dv*sc


def project_con(center, R_cw, con_intr, xyz):
    tvec = -R_cw @ center
    Xc   = R_cw @ np.asarray(xyz, dtype=np.float64) + tvec
    xb, yb, zb = Xc
    if zb <= 1e-9:
        return None
    f  = con_intr["focal"]
    u0 = con_intr["ppa_c"] + f * (xb / zb)
    v0 = con_intr["ppa_l"] + f * (yb / zb)
    return _con_distort(u0, v0, con_intr)


# ---------------------------------------------------------------------------
# Résidus
# ---------------------------------------------------------------------------

def compute_residuals(eors_df, tp3d, by_obs, con_intr,
                       rot_fn, flip, obs_fn, max_pts_per_image=0):
    per_image = []
    all_dc, all_dl, all_err = [], [], []

    for row in eors_df.itertuples(index=False):
        stem     = Path(str(row.label)).stem
        obs_path = by_obs.get(stem)
        if obs_path is None:
            continue
        obs_rows = parse_obs_file(obs_path, max_pts=max_pts_per_image)
        if not obs_rows:
            continue

        center = np.array([float(row.X0), float(row.Y0), float(row.Z0)],
                           dtype=np.float64)
        R_cw = build_R_cw(row.omega_deg, row.phi_deg, row.kappa_deg, rot_fn, flip)

        dc_img, dl_img, err_img = [], [], []
        n_behind = n_missing = 0

        for pid, c_obs, l_obs in obs_rows:
            xyz = tp3d.get(pid)
            if xyz is None:
                n_missing += 1; continue
            proj = project_con(center, R_cw, con_intr, xyz)
            if proj is None:
                n_behind += 1; continue
            u_ref, v_ref = obs_fn(float(c_obs), float(l_obs))
            dc = u_ref - proj[0];  dl = v_ref - proj[1]
            dc_img.append(dc);  dl_img.append(dl)
            err_img.append(math.sqrt(dc*dc + dl*dl))

        entry = {"stem": stem, "n_obs": len(obs_rows),
                 "n_valid": len(err_img),
                 "n_behind": n_behind, "n_missing": n_missing}
        if err_img:
            dc_a = np.asarray(dc_img); dl_a = np.asarray(dl_img)
            err_a = np.asarray(err_img)
            entry.update({
                "rmse":     float(np.sqrt(np.mean(err_a**2))),
                "mean_err": float(np.mean(err_a)),
                "max_err":  float(np.max(err_a)),
                "mean_dc":  float(np.mean(dc_a)),
                "mean_dl":  float(np.mean(dl_a)),
            })
            all_dc.extend(dc_img); all_dl.extend(dl_img); all_err.extend(err_img)
        else:
            entry.update({"rmse": None, "mean_err": None,
                          "max_err": None, "mean_dc": None, "mean_dl": None})
        per_image.append(entry)

    global_stats = None
    if all_err:
        ea = np.asarray(all_err); da = np.asarray(all_dc); la = np.asarray(all_dl)
        global_stats = {
            "n":        len(ea),
            "rmse":     float(np.sqrt(np.mean(ea**2))),
            "mean_err": float(np.mean(ea)),
            "med_err":  float(np.median(ea)),
            "p95_err":  float(np.percentile(ea, 95)),
            "max_err":  float(np.max(ea)),
            "mean_dc":  float(np.mean(da)),
            "mean_dl":  float(np.mean(la)),
        }
    return per_image, global_stats


def print_residuals(per_image, global_stats, title="RÉSIDUS"):
    W = 78
    print("\n" + "=" * W)
    print(title)
    print("=" * W)
    print(f"{'Image':<38} {'valid':>6} {'rmse':>7} {'mean':>7} "
          f"{'max':>7} {'dc':>8} {'dl':>8}")
    print("-" * W)
    for im in per_image:
        valid_str = f"{im['n_valid']}/{im['n_obs']}"
        if im["n_valid"] == 0:
            print(f"{im['stem']:<38} {valid_str:>6}  "
                  f"— (behind={im['n_behind']} missing={im['n_missing']})")
        else:
            print(f"{im['stem']:<38} {valid_str:>6} "
                  f"{im['rmse']:>7.3f} {im['mean_err']:>7.3f} {im['max_err']:>7.3f} "
                  f"{im['mean_dc']:>+8.3f} {im['mean_dl']:>+8.3f}")
    print("=" * W)
    if global_stats:
        g = global_stats
        print(f"GLOBAL  n={g['n']}  rmse={g['rmse']:.3f}px  "
              f"mean={g['mean_err']:.3f}px  med={g['med_err']:.3f}px  "
              f"p95={g['p95_err']:.3f}px  max={g['max_err']:.3f}px  "
              f"mean_dc={g['mean_dc']:+.3f}px  mean_dl={g['mean_dl']:+.3f}px")
    else:
        print("GLOBAL  aucune observation valide")
    print("=" * W + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Génère des .CON depuis EORS + _obs.txt avec calibration interne."
    )
    ap.add_argument("--eors",              required=True)
    ap.add_argument("--tp3d",              required=True)
    ap.add_argument("--images",            required=True)
    ap.add_argument("--out-dir",           default=None)
    ap.add_argument("--pixel-size",        type=float, default=4.52e-6)
    ap.add_argument("--geodesic",          default="LAMBERT93")
    ap.add_argument("--reference-label",   default=None)
    ap.add_argument("--rmse-threshold",    type=float, default=100.0)
    ap.add_argument("--pps-search-radius", type=float, default=300.0)
    ap.add_argument("--pps-search-step",   type=float, default=20.0)
    ap.add_argument("--max-pts-per-image", type=int,   default=0)
    ap.add_argument("--search-max-pts",    type=int,   default=50)
    ap.add_argument("--no-distortion",     action="store_true",
                    help="Ne fit pas la distorsion, écrit les .CON pinhole seul.")
    ap.add_argument("--verify-only",       action="store_true")
    args = ap.parse_args()

    eors_path  = Path(args.eors)
    tp3d_path  = Path(args.tp3d)
    images_dir = Path(args.images)
    out_dir    = Path(args.out_dir) if args.out_dir else None

    print(f"[INFO] Lecture EORS : {eors_path}")
    eors_df = load_eors(eors_path)
    print(f"[INFO] {len(eors_df)} orientations chargées")

    print(f"[INFO] Lecture tp3d : {tp3d_path}")
    tp3d = load_tp3d(tp3d_path)
    print(f"[INFO] {len(tp3d)} points 3D chargés")

    print(f"[INFO] Scan récursif sous : {images_dir}")
    by_image, by_obs = build_file_index(images_dir)
    n_ignored = sum(1 for r in eors_df.itertuples(index=False)
                    if Path(str(r.label)).stem not in by_image)
    print(f"[INFO] {len(by_image)} images, {len(by_obs)} _obs.txt, "
          f"{n_ignored} images EORS ignorées")

    ref_row        = choose_reference_row(eors_df, by_image, args.reference_label)
    ref_stem       = Path(str(ref_row.label)).stem
    ref_image_path = by_image[ref_stem]
    print(f"[INFO] Image de référence : {ref_image_path}")
    width, height  = get_image_size(ref_image_path)
    print(f"[INFO] Taille image : {width}x{height}")

    # 1. Recherche convention (avec obs(c, h-l) dans les candidats)
    print(f"\n[INFO] Recherche convention "
          f"({len(ROTATION_ORDERS)*len(FLIPS)*len(make_obs_convs(height))} hypothèses)...")
    rot_fn, flip, obs_fn, rot_name, flip_name, obs_name, _ = \
        search_best_convention(
            eors_df, tp3d, by_obs, height,
            search_max_pts=args.search_max_pts,
            rmse_threshold=args.rmse_threshold,
        )

    # 2. Pinhole complet
    print(f"\n[INFO] Fit pinhole complet ({rot_name}+{flip_name}+{obs_name})...")
    calib_data = build_calib_dataset(
        eors_df, tp3d, by_obs, rot_fn, flip, obs_fn,
        max_pts_per_image=args.max_pts_per_image,
    )
    if calib_data is None:
        raise RuntimeError("Aucune observation exploitable.")
    print(f"[INFO] {len(calib_data)} observations")

    f, ppa_c, ppa_l, rmse_pinhole = fit_pinhole(calib_data)
    print(f"[INFO] Pinhole : f={f:.4f}  ppa=({ppa_c:.4f},{ppa_l:.4f})  "
          f"rmse={rmse_pinhole:.4f}px")

    if f <= 0:
        raise RuntimeError(
            f"Focale négative (f={f:.2f}) après normalisation.\n"
            f"Convention '{rot_name}+{flip_name}+{obs_name}' incohérente."
        )

    con_pinhole = {
        "width": width, "height": height,
        "focal": f, "ppa_c": ppa_c, "ppa_l": ppa_l,
        "pps_c": ppa_c, "pps_l": ppa_l,
        "r1": 0.0, "r3": 0.0, "r5": 0.0, "r7": 0.0,
    }
    per_img_ph, gs_ph = compute_residuals(
        eors_df, tp3d, by_obs, con_pinhole, rot_fn, flip, obs_fn,
        max_pts_per_image=args.max_pts_per_image,
    )
    print_residuals(per_img_ph, gs_ph, "RÉSIDUS PINHOLE (sans distorsion)")

    if args.no_distortion:
        con_intr = con_pinhole
        print("[INFO] --no-distortion : pas de fit distorsion.")
    else:
        # 3. PPS
        print(f"[INFO] Recherche PPS "
              f"(rayon={args.pps_search_radius}px, pas={args.pps_search_step}px)...")
        pps_c, pps_l, score = fit_pps_grid(
            calib_data, f, ppa_c, ppa_l,
            search_radius=args.pps_search_radius,
            step=args.pps_search_step,
        )
        pps_c, pps_l = refine_pps(calib_data, f, ppa_c, ppa_l, pps_c, pps_l)
        print(f"[INFO] PPS : ({pps_c:.4f}, {pps_l:.4f})  "
              f"offset=({pps_c-ppa_c:+.2f}, {pps_l-ppa_l:+.2f})px")

        # 4. Distorsion
        print("[INFO] Fit distorsion radiale (r1, r3, r5, r7)...")
        r1, r3, r5, r7 = fit_distortion(calib_data, f, ppa_c, ppa_l, pps_c, pps_l)
        print(f"[INFO] r1={r1:.6e}  r3={r3:.6e}  r5={r5:.6e}  r7={r7:.6e}")

        con_intr = {
            "width": width, "height": height,
            "focal": f, "ppa_c": ppa_c, "ppa_l": ppa_l,
            "pps_c": pps_c, "pps_l": pps_l,
            "r1": r1, "r3": r3, "r5": r5, "r7": r7,
        }

        per_img, gs = compute_residuals(
            eors_df, tp3d, by_obs, con_intr, rot_fn, flip, obs_fn,
            max_pts_per_image=args.max_pts_per_image,
        )
        print_residuals(per_img, gs, "RÉSIDUS FINAUX (avec distorsion)")

    # 5. Écriture .CON
    if not args.verify_only:
        pixel_size_value = f"{float(args.pixel_size):.15e}"
        n_ok = n_skip = 0
        for row in eors_df.itertuples(index=False):
            stem       = Path(str(row.label)).stem
            image_path = by_image.get(stem)
            if image_path is None:
                n_skip += 1; continue

            center = np.array([float(row.X0), float(row.Y0), float(row.Z0)],
                               dtype=np.float64)
            R_cw   = build_R_cw(row.omega_deg, row.phi_deg, row.kappa_deg,
                                 rot_fn, flip)
            dest = out_dir / f"{stem}.CON" if out_dir is not None \
                else image_path.parent / f"{stem}.CON"
            dest.parent.mkdir(parents=True, exist_ok=True)

            root = build_orientation_xml(
                image_name=stem,
                center=center,
                R_cw=R_cw,
                con_intr=con_intr,
                geodesic_name=args.geodesic,
                pixel_size_value=pixel_size_value,
            )
            with open(dest, "wb") as fh:
                fh.write(prettify_xml(root))

            rel = dest.relative_to(images_dir) if out_dir is None else dest.name
            print(f"[INFO] {rel} : "
                  f"center=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f})")
            n_ok += 1

        print(f"\n[INFO] Terminé : {n_ok} .CON écrits, {n_skip} images ignorées.")


if __name__ == "__main__":
    main()

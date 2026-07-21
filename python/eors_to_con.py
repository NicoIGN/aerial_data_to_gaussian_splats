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

def make_obs_convs(height: int):
    h = float(height)
    return {
        "obs(c,h-l)": lambda c, l: (c, h - l),
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
# Fit pinhole — avec vérification PPA dans l'image
# ---------------------------------------------------------------------------

def fit_pinhole(data, width=None, height=None):
    xn, yn, u_obs, v_obs = data[:,0], data[:,1], data[:,2], data[:,3]
    N = len(xn)
    A = np.zeros((2*N, 3), dtype=np.float64)
    b = np.zeros(2*N, dtype=np.float64)
    A[:N,0]=1.0; A[:N,2]=xn; b[:N]=u_obs
    A[N:,1]=1.0; A[N:,2]=yn; b[N:]=v_obs
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    ppa_c, ppa_l, f = float(x[0]), float(x[1]), float(x[2])
    res_u = u_obs - (ppa_c + f*xn)
    res_v = v_obs - (ppa_l + f*yn)
    rmse  = float(np.sqrt(np.mean(res_u**2 + res_v**2)))

    # Vérification cohérence
    valid = f > 0
    if width  is not None: valid = valid and (0 <= ppa_c <= width)
    if height is not None: valid = valid and (0 <= ppa_l <= height)

    return f, ppa_c, ppa_l, rmse, valid


# ---------------------------------------------------------------------------
# Recherche convention — ne garde que les conventions avec PPA dans l'image
# ---------------------------------------------------------------------------

def search_best_convention(eors_df, tp3d, by_obs, width, height,
                            search_max_pts=50, rmse_threshold=100.0):
    obs_convs = make_obs_convs(height)
    results = []
    total = len(ROTATION_ORDERS) * len(FLIPS) * len(obs_convs)
    done  = 0

    for rot_name, rot_fn in ROTATION_ORDERS.items():
        for flip_name, flip_mat in FLIPS.items():
            for obs_name, obs_fn in obs_convs.items():
                data = build_calib_dataset(
                    eors_df, tp3d, by_obs, rot_fn, flip_mat, obs_fn,
                    max_pts_per_image=search_max_pts,
                )
                done += 1
                rmse = float("inf")
                if data is not None and len(data) >= 10:
                    f_val, pc, pl, rmse, valid = fit_pinhole(data, width, height)
                    if not valid:
                        rmse = float("inf")
                results.append((rmse, rot_name, flip_name, obs_name))
                best = min(results)
                print(f"\r  {done}/{total}  best: "
                      f"{best[1]}+{best[2]}+{best[3]} "
                      f"rmse={best[0]:.2f}px    ",
                      end="", flush=True)

    print()
    results.sort()

    print("\n" + "=" * 65)
    print("CLASSEMENT PINHOLE (top 10, f>0 et PPA dans l'image)")
    print("=" * 65)
    print(f"{'#':>3}  {'rotation':>10}  {'flip':>8}  {'obs':>12}  {'rmse':>8}")
    print("-" * 65)
    for i, (rmse, rot, flip, obs) in enumerate(results[:10], 1):
        print(f"{i:>3}  {rot:>10}  {flip:>8}  {obs:>12}  {rmse:>8.3f}px")
    print("=" * 65)

    best_rmse, best_rot, best_flip, best_obs = results[0]
    print(f"\n[INFO] Meilleure convention : "
          f"rotation={best_rot}  flip={best_flip}  obs={best_obs}  "
          f"rmse={best_rmse:.3f}px")

    if best_rmse > rmse_threshold:
        raise RuntimeError(
            f"RMSE pinhole = {best_rmse:.2f}px > seuil {rmse_threshold}px.\n"
            f"Aucune convention valide (f>0, PPA dans image) trouvée.\n"
            f"Vérifiez le système de coordonnées ou la qualité du tp3d."
        )

    obs_convs_full = make_obs_convs(height)
    return (ROTATION_ORDERS[best_rot], FLIPS[best_flip],
            obs_convs_full[best_obs],
            best_rot, best_flip, best_obs, best_rmse)



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
            dc = u_ref - proj[0]; dl = v_ref - proj[1]
            dc_img.append(dc); dl_img.append(dl)
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

def fit_all_intrinsics(data, f0, ppa_c0, ppa_l0,
                        pps_search_radius=300.0, pps_search_step=10.0,
                        gn_iters=50, verbose=True):
    """
    Fit séquencé :
      1. PPS par critère tangentiel (grille + GN 2 params), f/ppa fixés
      2. Fit linéaire r3,r5,r7 (r1=0, PPS fixé)
      3. Fit linéaire r1 seul (r3,r5,r7 fixés)
      4. GN conjoint 9 params [f, ppa_c, ppa_l, pps_dc, pps_dl, r1, r3, r5, r7]
    """
    xn, yn, u_obs, v_obs = data[:,0], data[:,1], data[:,2], data[:,3]
    N = len(xn)

    # ------------------------------------------------------------------
    # Étape 1 : PPS par critère tangentiel
    # ------------------------------------------------------------------
    u0    = ppa_c0 + f0 * xn
    v0    = ppa_l0 + f0 * yn
    ru_ph = u_obs - u0
    rv_ph = v_obs - v0

    offsets = np.arange(-pps_search_radius,
                         pps_search_radius + pps_search_step*0.5,
                         pps_search_step)
    best_score = float("inf")
    best_dc, best_dl = 0.0, 0.0
    for dc in offsets:
        for dl in offsets:
            du = u0 - (ppa_c0 + dc);  dv = v0 - (ppa_l0 + dl)
            R  = np.sqrt(du**2 + dv**2)
            valid = R > 1.0
            if not np.any(valid):
                continue
            tang  = (-dv[valid]*ru_ph[valid] + du[valid]*rv_ph[valid]) / R[valid]
            score = float(np.mean(tang**2))
            if score < best_score:
                best_score = score;  best_dc, best_dl = dc, dl

    pps_c = ppa_c0 + best_dc;  pps_l = ppa_l0 + best_dl
    if verbose:
        print(f"  [1-PPS grille]  offset=({best_dc:+.1f},{best_dl:+.1f})px")

    for _ in range(50):
        du = u0 - pps_c;  dv = v0 - pps_l
        R  = np.sqrt(du**2 + dv**2);  valid = R > 1.0
        tang = (-dv[valid]*ru_ph[valid] + du[valid]*rv_ph[valid]) / R[valid]
        duv = du[valid];  dvv = dv[valid];  Rv = R[valid]
        J_pc = ( dvv*ru_ph[valid] - duv*rv_ph[valid]) / Rv
        J_pl = (-dvv*rv_ph[valid] - duv*ru_ph[valid]) / Rv
        JtJ  = np.array([[np.sum(J_pc**2),   np.sum(J_pc*J_pl)],
                          [np.sum(J_pc*J_pl), np.sum(J_pl**2)  ]])
        Jtr  = np.array([np.sum(J_pc*tang), np.sum(J_pl*tang)])
        try:
            delta = np.linalg.solve(JtJ + 1e-8*np.eye(2), Jtr)
        except np.linalg.LinAlgError:
            break
        if np.linalg.norm(delta) < 1e-4:
            break
        pps_c -= delta[0];  pps_l -= delta[1]

    if verbose:
        print(f"  [1-PPS affiné]  pps=({pps_c:.3f},{pps_l:.3f})  "
              f"offset=({pps_c-ppa_c0:+.3f},{pps_l-ppa_l0:+.3f})px")

    # ------------------------------------------------------------------
    # Étape 2 : fit linéaire r3,r5,r7 (r1=0, PPS fixé)
    # ------------------------------------------------------------------
    du = u0 - pps_c;  dv = v0 - pps_l
    R2 = (du**2 + dv**2) / (f0*f0)

    rhs_u = (u_obs - pps_c) - du
    rhs_v = (v_obs - pps_l) - dv
    A2 = np.zeros((2*N, 3), dtype=np.float64)
    b2 = np.zeros(2*N, dtype=np.float64)
    for j, power in enumerate([2, 3, 4]):
        Rp = R2**power
        A2[:N,j] = du*Rp;  A2[N:,j] = dv*Rp
    b2[:N] = rhs_u;  b2[N:] = rhs_v
    c357, *_ = np.linalg.lstsq(A2, b2, rcond=None)
    r3, r5, r7 = float(c357[0]), float(c357[1]), float(c357[2])
    if verbose:
        print(f"  [2-r3,r5,r7]    r3={r3:.4e}  r5={r5:.4e}  r7={r7:.4e}")

    # ------------------------------------------------------------------
    # Étape 3 : fit linéaire r1 seul (r3,r5,r7 fixés)
    # résidu après r3,r5,r7 ≈ du*r1*R2  +  dv*r1*R2
    # ------------------------------------------------------------------
    scale_357   = r3*R2**2 + r5*R2**3 + r7*R2**4
    u_after_357 = pps_c + du*(1.0 + scale_357)
    v_after_357 = pps_l + dv*(1.0 + scale_357)
    resid_u = u_obs - u_after_357
    resid_v = v_obs - v_after_357
    A1  = np.concatenate([du*R2, dv*R2])
    b1  = np.concatenate([resid_u, resid_v])
    r1  = float(np.dot(A1, b1) / (np.dot(A1, A1) + 1e-12))

    if verbose:
        scale_all = 1.0 + r1*R2 + r3*R2**2 + r5*R2**3 + r7*R2**4
        u_p = pps_c + du*scale_all;  v_p = pps_l + dv*scale_all
        err = np.sqrt((u_obs-u_p)**2 + (v_obs-v_p)**2)
        print(f"  [3-r1]          r1={r1:.4e}  "
              f"rmse après init={np.sqrt(np.mean(err**2)):.4f}px")

    # ------------------------------------------------------------------
    # Étape 4 : GN conjoint 9 paramètres
    # theta = [f, ppa_c, ppa_l, pps_dc, pps_dl, r1, r3, r5, r7]
    # ------------------------------------------------------------------
    theta = np.array([f0, ppa_c0, ppa_l0,
                       pps_c - ppa_c0, pps_l - ppa_l0,
                       r1, r3, r5, r7], dtype=np.float64)

    def _pred(th):
        _f   = th[0];  _pc  = th[1];  _pl  = th[2]
        _dc  = th[3];  _dl  = th[4]
        _r1  = th[5];  _r3  = th[6];  _r5  = th[7];  _r7 = th[8]
        _pps_c = _pc + _dc;  _pps_l = _pl + _dl
        _u0  = _pc + _f * xn;  _v0 = _pl + _f * yn
        _du  = _u0 - _pps_c;   _dv = _v0 - _pps_l
        _R2  = (_du**2 + _dv**2) / (_f*_f)
        _sc  = 1.0 + _r1*_R2 + _r3*_R2**2 + _r5*_R2**3 + _r7*_R2**4
        return _pps_c + _du*_sc, _pps_l + _dv*_sc

    def _res(th):
        up, vp = _pred(th)
        return np.concatenate([u_obs - up, v_obs - vp])

    best_theta = theta.copy()
    best_rmse  = float("inf")
    prev_rmse  = float("inf")
    eps = 1e-5

    for it in range(gn_iters):
        res  = _res(theta)
        rmse = float(np.sqrt(np.mean(res**2)))

        if rmse < best_rmse:
            best_rmse = rmse;  best_theta = theta.copy()

        if verbose:
            print(f"  [4-GN iter {it:2d}] rmse={rmse:.4f}px  "
                  f"f={theta[0]:.2f}  "
                  f"ppa=({theta[1]:.1f},{theta[2]:.1f})  "
                  f"pps_d=({theta[3]:+.1f},{theta[4]:+.1f})  "
                  f"r1={theta[5]:.3e}  r3={theta[6]:.3e}")

        if abs(prev_rmse - rmse) < 1e-6:
            if verbose:
                print(f"  [4-GN converged iter {it}]")
            break
        prev_rmse = rmse

        # Jacobienne diff. finies
        npar = len(theta)
        J = np.zeros((2*N, npar), dtype=np.float64)
        for j in range(npar):
            tp = theta.copy(); tp[j] += eps
            tm = theta.copy(); tm[j] -= eps
            up_p, vp_p = _pred(tp);  up_m, vp_m = _pred(tm)
            J[:N, j] = (up_p - up_m) / (2*eps)
            J[N:, j] = (vp_p - vp_m) / (2*eps)

        JtJ = J.T @ J
        Jtr = J.T @ res   # résidu = obs - pred

        # Régularisation : quasi-nulle sur géométrie, légère sur distorsion
        reg = np.array([1e-8, 1e-8, 1e-8,   # f, ppa_c, ppa_l
                         1e-8, 1e-8,          # pps_dc, pps_dl
                         1e-6, 1e-6, 1e-6, 1e-6])  # r1,r3,r5,r7
        JtJ += np.diag(reg)

        try:
            delta = np.linalg.solve(JtJ, Jtr)
        except np.linalg.LinAlgError:
            break

        # Pas limité par paramètre
        max_steps = np.array([50.0, 20.0, 20.0,   # f, ppa_c, ppa_l
                               20.0, 20.0,          # pps_dc, pps_dl
                               1e-2, 1e-2, 1e-2, 1e-2])  # ri
        delta = np.clip(delta, -max_steps, max_steps)

        theta = theta + delta

    theta = best_theta
    up, vp = _pred(theta)
    err = np.sqrt((u_obs-up)**2 + (v_obs-vp)**2)
    stats = {
        "rmse":     float(np.sqrt(np.mean(err**2))),
        "mean_err": float(np.mean(err)),
        "med_err":  float(np.median(err)),
        "p95_err":  float(np.percentile(err, 95)),
        "max_err":  float(np.max(err)),
        "n":        N,
    }

    f_out    = float(theta[0])
    ppa_c_out = float(theta[1]);  ppa_l_out = float(theta[2])
    pps_c_out = float(theta[1] + theta[3])
    pps_l_out = float(theta[2] + theta[4])
    r1_out, r3_out, r5_out, r7_out = (float(theta[5]), float(theta[6]),
                                       float(theta[7]), float(theta[8]))

    if verbose:
        print(f"\n  [résultat final]")
        print(f"    f={f_out:.4f}  ppa=({ppa_c_out:.3f},{ppa_l_out:.3f})")
        print(f"    pps=({pps_c_out:.3f},{pps_l_out:.3f})  "
              f"offset=({pps_c_out-ppa_c_out:+.3f},{pps_l_out-ppa_l_out:+.3f})px")
        print(f"    r1={r1_out:.6e}  r3={r3_out:.6e}  "
              f"r5={r5_out:.6e}  r7={r7_out:.6e}")
        print(f"    rmse={stats['rmse']:.4f}px  p95={stats['p95_err']:.4f}px  "
              f"max={stats['max_err']:.4f}px")

    return f_out, ppa_c_out, ppa_l_out, pps_c_out, pps_l_out, \
           r1_out, r3_out, r5_out, r7_out, stats

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
    ap.add_argument("--max-pts-per-image", type=int,   default=0)
    ap.add_argument("--search-max-pts",    type=int,   default=50)
    ap.add_argument("--no-distortion",     action="store_true")
    ap.add_argument("--verify-only",       action="store_true")
    ap.add_argument("--pps-search-radius", type=float, default=300.0)
    ap.add_argument("--pps-search-step",   type=float, default=10.0)
    ap.add_argument("--gn-iters",          type=int,   default=50)
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

    # 1. Recherche convention (f>0, PPA dans l'image)
    print(f"\n[INFO] Recherche convention "
          f"({len(ROTATION_ORDERS)*len(FLIPS)*len(make_obs_convs(height))} hypothèses)...")
    rot_fn, flip, obs_fn, rot_name, flip_name, obs_name, _ = \
        search_best_convention(
            eors_df, tp3d, by_obs, width, height,
            search_max_pts=args.search_max_pts,
            rmse_threshold=args.rmse_threshold,
        )

    # 2. Pinhole complet
    print(f"\n[INFO] Fit pinhole ({rot_name}+{flip_name}+{obs_name})...")
    calib_data = build_calib_dataset(
        eors_df, tp3d, by_obs, rot_fn, flip, obs_fn,
        max_pts_per_image=args.max_pts_per_image,
    )
    if calib_data is None:
        raise RuntimeError("Aucune observation exploitable.")
    print(f"[INFO] {len(calib_data)} observations")

    f, ppa_c, ppa_l, rmse_ph, valid = fit_pinhole(calib_data, width, height)
    print(f"[INFO] Pinhole : f={f:.4f}  ppa=({ppa_c:.4f},{ppa_l:.4f})  "
          f"rmse={rmse_ph:.4f}px  valid={valid}")

    if not valid:
        raise RuntimeError(
            f"Pinhole invalide : f={f:.2f}, ppa=({ppa_c:.1f},{ppa_l:.1f})\n"
            f"PPA hors image ou focale négative."
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
    else:
        print(f"\n[INFO] Calibration complète "
              f"(PPA+PPS+r1+r3+r5+r7, {args.gn_iters} iters GN)...")
        f_cal, ppa_c, ppa_l, pps_c, pps_l, r1, r3, r5, r7, stats = \
            fit_all_intrinsics(
                calib_data, f, ppa_c, ppa_l,
                pps_search_radius=args.pps_search_radius,
                pps_search_step=args.pps_search_step,
                gn_iters=args.gn_iters,
                verbose=True,
            )
        print(f"[INFO] Résidus fit : "
              f"rmse={stats['rmse']:.4f}px  p95={stats['p95_err']:.4f}px  "
              f"max={stats['max_err']:.4f}px  n={stats['n']}")

        con_intr = {
            "width": width, "height": height,
            "focal": f_cal, "ppa_c": ppa_c, "ppa_l": ppa_l,
            "pps_c": pps_c, "pps_l": pps_l,
            "r1": r1, "r3": r3, "r5": r5, "r7": r7,
        }

        per_img, gs = compute_residuals(
            eors_df, tp3d, by_obs, con_intr, rot_fn, flip, obs_fn,
            max_pts_per_image=args.max_pts_per_image,
        )
        print_residuals(per_img, gs, "RÉSIDUS FINAUX (avec distorsion)")

    # 4. Écriture .CON
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
            R_cw = build_R_cw(row.omega_deg, row.phi_deg, row.kappa_deg,
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

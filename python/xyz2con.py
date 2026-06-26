#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ========================= maths =========================

def d2r(v): return v * math.pi / 180.0

def Rx(a):
    ca, sa = math.cos(a), math.sin(a)
    return [[1, 0, 0], [0, ca, -sa], [0, sa, ca]]

def Ry(a):
    ca, sa = math.cos(a), math.sin(a)
    return [[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]]

def Rz(a):
    ca, sa = math.cos(a), math.sin(a)
    return [[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]]

def mm(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def transpose3(R):
    return [[R[0][0], R[1][0], R[2][0]],
            [R[0][1], R[1][1], R[2][1]],
            [R[0][2], R[1][2], R[2][2]]]

def mat_vec_mul(R, v):
    return [
        R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
        R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
        R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2],
    ]

def opk_to_R(omega_deg, phi_deg, kappa_deg, order="RxRyRz"):
    o, p, k = d2r(omega_deg), d2r(phi_deg), d2r(kappa_deg)
    if order == "RxRyRz":
        return mm(mm(Rx(o), Ry(p)), Rz(k))
    if order == "RzRyRx":
        return mm(mm(Rz(k), Ry(p)), Rx(o))
    raise ValueError("order must be RxRyRz or RzRyRx")

def apply_conv(R, conv="rx180"):
    if conv == "none":
        C = [[1,0,0],[0,1,0],[0,0,1]]
    elif conv == "rx180":
        C = [[1,0,0],[0,-1,0],[0,0,-1]]
    elif conv == "ry180":
        C = [[-1,0,0],[0,1,0],[0,0,-1]]
    elif conv == "rz180":
        C = [[-1,0,0],[0,-1,0],[0,0,1]]
    else:
        raise ValueError("conv invalide")
    return mm(R, C)


# ========================= IO =========================

def parse_obs_file(obs_path: Path):
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
                pid = int(float(p[0]))
                u = float(p[1])
                v = -float(p[2])
                rows.append((pid, u, v))
            except Exception:
                pass
    return rows

def load_tp3d(tp_path: Path):
    d = {}
    with open(tp_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.replace("\t", " ").split()
            if len(p) < 4:
                continue
            try:
                pid = int(float(p[0]))
                d[pid] = (float(p[1]), float(p[2]), float(p[3]))
            except Exception:
                pass
    return d

def load_colmap_camera_txt(camera_txt_path: Path):
    """
    Parse cameras.txt COLMAP.
    Retourne dict:
      {
        "camera_id": int,
        "model": str,
        "width": int,
        "height": int,
        "params": [float...]
      }
    """
    cam = None
    with open(camera_txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            # CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
            if len(p) < 5:
                continue
            cam = {
                "camera_id": int(p[0]),
                "model": p[1],
                "width": int(p[2]),
                "height": int(p[3]),
                "params": [float(x) for x in p[4:]]
            }
            break
    if cam is None:
        raise RuntimeError(f"Aucune caméra valide trouvée dans {camera_txt_path}")
    return cam

def camera_params_to_intrinsics(cam):
    """
    Support minimal: OPENCV
    OPENCV params: fx fy cx cy k1 k2 p1 p2
    """
    model = cam["model"].upper()
    prm = cam["params"]

    if model == "OPENCV":
        if len(prm) < 8:
            raise ValueError("MODEL OPENCV requiert 8 params: fx fy cx cy k1 k2 p1 p2")
        fx, fy, cx, cy, k1, k2, p1, p2 = prm[:8]
        return {
            "model": model,
            "fx": fx, "fy": fy,
            "cx": cx, "cy": cy,
            "k1": k1, "k2": k2, "p1": p1, "p2": p2,
            "width": cam["width"], "height": cam["height"]
        }
    else:
        raise NotImplementedError(f"Model non supporté pour l'instant: {cam['model']}")


# ========================= projection / stats =========================

def distort_radial(u, v, cx, cy, width, height, k1=0.0):
    norm = max(1.0, 0.5 * math.hypot(width, height))
    x = (u - cx) / norm
    y = (v - cy) / norm
    r2 = x*x + y*y
    s = 1.0 + k1*r2
    return cx + (x*s)*norm, cy + (y*s)*norm

def project_point(
    Xw, Yw, Zw, Cx, Cy, Cz, R_img2ground,
    ppa_c, ppa_l, focale, width, height,
    image2ground=True, z_positive=True, k1=0.0
):
    Rcw = transpose3(R_img2ground) if image2ground else R_img2ground
    d = [Xw - Cx, Yw - Cy, Zw - Cz]
    Xc, Yc, Zc = mat_vec_mul(Rcw, d)

    if z_positive:
        if Zc <= 1e-9: return None
        z = Zc
    else:
        if Zc >= -1e-9: return None
        z = -Zc

    u = ppa_c + focale * (Xc / z)
    v = ppa_l + focale * (Yc / z)
    if k1 != 0.0:
        u, v = distort_radial(u, v, ppa_c, ppa_l, width, height, k1=k1)
    return u, v

def robust_stats(errors):
    if not errors:
        return {"n": 0, "med": np.nan, "rmse": np.nan, "p95": np.nan}
    e = np.asarray(errors, dtype=np.float64)
    return {
        "n": int(len(e)),
        "med": float(np.median(e)),
        "rmse": float(np.sqrt(np.mean(e*e))),
        "p95": float(np.percentile(e, 95)),
    }

def score_from_stats(med, rmse, p95, behind):
    return 0.35*med + 0.45*rmse + 0.20*p95 + 1e6*behind

def make_grid(center, radius, steps):
    if steps <= 1:
        return np.array([center], dtype=float)
    return np.linspace(center - radius, center + radius, steps)


# ========================= sampling / eval =========================

def build_samples(df, images_dir, obs_dir, y0_sign, fit_stride, fit_max_images):
    sub = df.iloc[::max(1, fit_stride)].copy()
    if fit_max_images > 0:
        sub = sub.head(fit_max_images)

    samples = []
    for _, row in sub.iterrows():
        stem = Path(str(row["label"])).stem
        jpg = images_dir / f"{stem}.jpg"
        obsf = obs_dir / f"{stem}_obs.txt"
        if not jpg.exists() or not obsf.exists():
            continue
        obs_rows = parse_obs_file(obsf)
        if not obs_rows:
            continue
        with Image.open(jpg) as im:
            width, height = im.size
        samples.append({
            "row": row,
            "obs_rows": obs_rows,
            "width": width,
            "height": height,
            "base_cx_center": width / 2.0,
            "base_cy_center": height / 2.0,
            "base_cx_input": float(row["x0"]),
            "base_cy_input": float(y0_sign) * float(row["y0"]),
        })
    return samples

def eval_hyp_with_fixed_center(samples, tp3d, cfg):
    meds, rmses, p95s = [], [], []
    total_behind, total_n = 0, 0

    for s in samples:
        row = s["row"]
        R = apply_conv(
            opk_to_R(float(row["omega[deg]"]), float(row["phi[deg]"]), float(row["kappa[deg]"]), order=cfg["order"]),
            cfg["conv_rot"]
        )
        errs, behind = [], 0
        for pid, u_gt, v_gt in s["obs_rows"]:
            xyz = tp3d.get(pid)
            if xyz is None: continue
            proj = project_point(
                xyz[0], xyz[1], xyz[2],
                float(row["X0"]), float(row["Y0"]), float(row["Z0"]),
                R, s["base_cx_center"], s["base_cy_center"], float(row["c"]),
                s["width"], s["height"],
                image2ground=cfg["image2ground"], z_positive=cfg["z_positive"], k1=0.0
            )
            if proj is None:
                behind += 1
                continue
            u, v = proj
            errs.append(math.hypot(u-u_gt, v-v_gt))

        st = robust_stats(errs)
        total_behind += behind
        total_n += st["n"]
        if st["n"] > 0:
            meds.append(st["med"]); rmses.append(st["rmse"]); p95s.append(st["p95"])

    if not meds:
        return {"score": 1e12, "med": np.nan, "rmse": np.nan, "p95": np.nan, "behind": total_behind, "n": total_n}
    med, rmse, p95 = float(np.mean(meds)), float(np.mean(rmses)), float(np.mean(p95s))
    return {"score": score_from_stats(med, rmse, p95, total_behind), "med": med, "rmse": rmse, "p95": p95, "behind": total_behind, "n": total_n}

def eval_params(samples, tp3d, cfg, dc, dl, k1, domega, dphi, use_input_ppa=True):
    meds, rmses, p95s = [], [], []
    total_behind, total_n = 0, 0

    for s in samples:
        row = s["row"]
        base_cx = s["base_cx_input"] if use_input_ppa else s["base_cx_center"]
        base_cy = s["base_cy_input"] if use_input_ppa else s["base_cy_center"]
        ppa_c, ppa_l = base_cx + dc, base_cy + dl

        om = float(row["omega[deg]"]) + domega
        ph = float(row["phi[deg]"]) + dphi
        ka = float(row["kappa[deg]"])

        R = apply_conv(opk_to_R(om, ph, ka, order=cfg["order"]), cfg["conv_rot"])

        errs, behind = [], 0
        for pid, u_gt, v_gt in s["obs_rows"]:
            xyz = tp3d.get(pid)
            if xyz is None: continue
            proj = project_point(
                xyz[0], xyz[1], xyz[2],
                float(row["X0"]), float(row["Y0"]), float(row["Z0"]),
                R, ppa_c, ppa_l, float(row["c"]),
                s["width"], s["height"],
                image2ground=cfg["image2ground"], z_positive=cfg["z_positive"], k1=k1
            )
            if proj is None:
                behind += 1
                continue
            u, v = proj
            errs.append(math.hypot(u-u_gt, v-v_gt))

        st = robust_stats(errs)
        total_behind += behind
        total_n += st["n"]
        if st["n"] > 0:
            meds.append(st["med"]); rmses.append(st["rmse"]); p95s.append(st["p95"])

    if not meds:
        return {"score": 1e12, "med": np.nan, "rmse": np.nan, "p95": np.nan, "behind": total_behind, "n": total_n}
    med, rmse, p95 = float(np.mean(meds)), float(np.mean(rmses)), float(np.mean(p95s))
    return {"score": score_from_stats(med, rmse, p95, total_behind), "med": med, "rmse": rmse, "p95": p95, "behind": total_behind, "n": total_n}


# ========================= optimization core =========================

def optimize_phase2(samples, tp3d, cfg, fit, use_input_ppa=True):
    center = {"dc": 0.0, "dl": 0.0, "k1": 0.0, "domega": 0.0, "dphi": 0.0}
    logs, stage_bests = [], {}
    best_global = None
    test_idx = 0

    stages = [
        ("coarse", fit["coarse_r"], fit["coarse_s"]),
        ("ref1", fit["ref1_r"], fit["ref1_s"]),
        ("ref2", fit["ref2_r"], fit["ref2_s"]),
    ]

    for stage_name, r, s in stages:
        dc_vals = make_grid(center["dc"], r["dc"], s["dc"])
        dl_vals = make_grid(center["dl"], r["dl"], s["dl"])
        k1_vals = make_grid(center["k1"], r["k1"], s["k1"])
        do_vals = make_grid(center["domega"], r["domega"], s["domega"])
        dp_vals = make_grid(center["dphi"], r["dphi"], s["dphi"])

        combos = list(itertools.product(dc_vals, dl_vals, k1_vals, do_vals, dp_vals))
        it = tqdm(combos, total=len(combos), desc=f"[FIT:{stage_name}]", unit="test") if tqdm else combos

        stage_rows, stage_best = [], None
        for dc, dl, k1, domega, dphi in it:
            test_idx += 1
            res = eval_params(samples, tp3d, cfg, dc, dl, k1, domega, dphi, use_input_ppa=use_input_ppa)
            row = {"stage": stage_name, "test_idx": test_idx, "dc": dc, "dl": dl, "k1": k1, "domega": domega, "dphi": dphi, **res}
            logs.append(row); stage_rows.append(row)

            if stage_best is None or row["score"] < stage_best["score"]:
                stage_best = row
            if best_global is None or row["score"] < best_global["score"]:
                best_global = row

        stage_bests[stage_name] = stage_best
        center = {"dc": stage_best["dc"], "dl": stage_best["dl"], "k1": stage_best["k1"], "domega": stage_best["domega"], "dphi": stage_best["dphi"]}

        top5 = pd.DataFrame(stage_rows).sort_values("score").head(5)
        print(f"[FIT:{stage_name}] top5:\n{top5[['dc','dl','k1','domega','dphi','med','rmse','p95','score']].to_string(index=False)}")
        print(f"[FIT:{stage_name}] best => dc={center['dc']:.3f} dl={center['dl']:.3f} k1={center['k1']:.6f} dω={center['domega']:.4f} dφ={center['dphi']:.4f} rmse={stage_best['rmse']:.2f}")

    return best_global, pd.DataFrame(logs), stage_bests

def optimize_domega_dphi_final(samples, tp3d, cfg, fixed_params, use_input_ppa=True):
    dc0, dl0, k10 = float(fixed_params["dc"]), float(fixed_params["dl"]), float(fixed_params["k1"])
    stages = [("ang_coarse", 0.30, 5), ("ang_fine", 0.08, 5)]

    best = {"domega": 0.0, "dphi": 0.0, "score": float("inf"), "med": np.nan, "rmse": np.nan, "p95": np.nan}
    logs, test_idx = [], 0

    for stage_name, radius, steps in stages:
        do_vals = make_grid(best["domega"], radius, steps)
        dp_vals = make_grid(best["dphi"], radius, steps)
        combos = list(itertools.product(do_vals, dp_vals))
        it = tqdm(combos, total=len(combos), desc=f"[FIT:{stage_name}]", unit="test") if tqdm else combos

        stage_rows, stage_best = [], None
        for domega, dphi in it:
            test_idx += 1
            res = eval_params(samples, tp3d, cfg, dc0, dl0, k10, domega, dphi, use_input_ppa=use_input_ppa)
            row = {"stage": stage_name, "test_idx": test_idx, "dc": dc0, "dl": dl0, "k1": k10, "domega": domega, "dphi": dphi, **res}
            logs.append(row); stage_rows.append(row)
            if stage_best is None or row["score"] < stage_best["score"]:
                stage_best = row
            if row["score"] < best["score"]:
                best = row

        top5 = pd.DataFrame(stage_rows).sort_values("score").head(5)
        print(f"[FIT:{stage_name}] top5:\n{top5[['domega','dphi','med','rmse','p95','score']].to_string(index=False)}")
        print(f"[FIT:{stage_name}] best => dω={stage_best['domega']:.4f} dφ={stage_best['dphi']:.4f} rmse={stage_best['rmse']:.2f}")

    return best, pd.DataFrame(logs)


# ========================= xml =========================

def indent(elem, level=0):
    i = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        for e in elem:
            indent(e, level + 1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

def fmt(v, n=12): return f"{float(v):.{n}f}"

def build_con(row, ppa_c, ppa_l, width, height, domega, dphi, order, conv_rot="rx180", image2ground=True, focale_override=None):
    R = apply_conv(
        opk_to_R(
            float(row["omega[deg]"]) + domega,
            float(row["phi[deg]"]) + dphi,
            float(row["kappa[deg]"]),
            order=order
        ),
        conv_rot
    )

    root = ET.Element("orientation")
    ET.SubElement(root, "version").text = "1.0"
    geom = ET.SubElement(root, "geometry", {"type": "physique"})
    ext = ET.SubElement(geom, "extrinseque")

    sys = ET.SubElement(ext, "systeme")
    euc = ET.SubElement(sys, "euclidien", {"type": "MATISRTL"})
    ET.SubElement(euc, "x").text = fmt(row["X0"])
    ET.SubElement(euc, "y").text = fmt(row["Y0"])

    sommet = ET.SubElement(ext, "sommet")
    ET.SubElement(sommet, "easting").text = "0"
    ET.SubElement(sommet, "northing").text = "0"
    ET.SubElement(sommet, "altitude").text = fmt(row["Z0"])

    rot = ET.SubElement(ext, "rotation")
    ET.SubElement(rot, "Image2Ground").text = "true" if image2ground else "false"
    m = ET.SubElement(rot, "mat3d")
    for i in range(3):
        li = ET.SubElement(m, f"l{i+1}")
        pt = ET.SubElement(li, "pt3d")
        ET.SubElement(pt, "x").text = fmt(R[i][0], 15)
        ET.SubElement(pt, "y").text = fmt(R[i][1], 15)
        ET.SubElement(pt, "z").text = fmt(R[i][2], 15)

    intr = ET.SubElement(geom, "intrinseque")
    sensor = ET.SubElement(intr, "sensor")

    image_size = ET.SubElement(sensor, "image_size")
    ET.SubElement(image_size, "width").text = str(int(width))
    ET.SubElement(image_size, "height").text = str(int(height))

    ppa = ET.SubElement(sensor, "ppa")
    ET.SubElement(ppa, "c").text = fmt(ppa_c)
    ET.SubElement(ppa, "l").text = fmt(ppa_l)
    foc = float(focale_override) if focale_override is not None else float(row["c"])
    ET.SubElement(ppa, "focale").text = fmt(foc)

    return ET.ElementTree(root), Path(str(row["label"])).stem


# ========================= 1) parse args =========================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--camera_txt", type=str, default=None,
                    help="Chemin vers cameras.txt COLMAP (intrinsics figées). Si fourni, on n'estime pas les params caméra.")
    ap.add_argument("--obs_dir", required=True)
    ap.add_argument("--tp3d", required=True)
    ap.add_argument("--report_csv", default=None)
    ap.add_argument("--fit_report_csv", default=None)
    ap.add_argument("--hyp_report_csv", default=None)
    ap.add_argument("--z_negative", action="store_true")
    ap.add_argument("--fit_stride", type=int, default=8)
    ap.add_argument("--fit_max_images", type=int, default=40)
    ap.add_argument("--use_input_ppa", action="store_true", default=True)
    ap.add_argument("--use_center_ppa", dest="use_input_ppa", action="store_false")

    ap.add_argument("--phase2_rel_abort", type=float, default=2.0)
    ap.add_argument("--phase2_abs_abort_rmse", type=float, default=200.0)

    ap.add_argument("--coarse_r_dc", type=float, default=120.0)
    ap.add_argument("--coarse_r_dl", type=float, default=120.0)
    ap.add_argument("--coarse_r_k1", type=float, default=0.08)
    ap.add_argument("--coarse_r_domega", type=float, default=0.8)
    ap.add_argument("--coarse_r_dphi", type=float, default=0.8)

    ap.add_argument("--ref1_r_dc", type=float, default=40.0)
    ap.add_argument("--ref1_r_dl", type=float, default=40.0)
    ap.add_argument("--ref1_r_k1", type=float, default=0.02)
    ap.add_argument("--ref1_r_domega", type=float, default=0.25)
    ap.add_argument("--ref1_r_dphi", type=float, default=0.25)

    ap.add_argument("--ref2_r_dc", type=float, default=16.0)
    ap.add_argument("--ref2_r_dl", type=float, default=16.0)
    ap.add_argument("--ref2_r_k1", type=float, default=0.008)
    ap.add_argument("--ref2_r_domega", type=float, default=0.08)
    ap.add_argument("--ref2_r_dphi", type=float, default=0.08)

    return ap.parse_args()


# ========================= 2) optimisation =========================

def run_optimisation(args):
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir); obs_dir = Path(args.obs_dir)
    tp3d = load_tp3d(Path(args.tp3d))
    print(f"[INFO] points 3D chargés: {len(tp3d)}")

    df = pd.read_csv(args.input, sep=r"\s+", engine="python")
    if "#label" in df.columns: df = df.rename(columns={"#label": "label"})
    if "intLabel" in df.columns: df = df.rename(columns={"intLabel": "label"})

    fixed_cam = None
    if args.camera_txt:
        fixed_cam_raw = load_colmap_camera_txt(Path(args.camera_txt))
        fixed_cam = camera_params_to_intrinsics(fixed_cam_raw)
        print(f"[INFO] camera_txt chargé: model={fixed_cam['model']} "
              f"{fixed_cam['width']}x{fixed_cam['height']} "
              f"fx={fixed_cam['fx']:.3f} fy={fixed_cam['fy']:.3f} "
              f"cx={fixed_cam['cx']:.3f} cy={fixed_cam['cy']:.3f} k1={fixed_cam['k1']:.6f}")

    # phase1 screening
    orders = ["RxRyRz", "RzRyRx"]
    convs = ["none", "rx180", "ry180", "rz180"]
    i2g_vals = [True, False]
    y0_vals = [-1, 1]

    hyp_rows = []
    for order, conv_rot, image2ground, y0_sign in itertools.product(orders, convs, i2g_vals, y0_vals):
        samples = build_samples(df, images_dir, obs_dir, y0_sign, args.fit_stride, args.fit_max_images)
        if not samples: continue
        cfg = {"order": order, "conv_rot": conv_rot, "image2ground": image2ground, "z_positive": (not args.z_negative)}
        st = eval_hyp_with_fixed_center(samples, tp3d, cfg)
        hyp_rows.append({"order": order, "conv_rot": conv_rot, "image2ground": image2ground, "y0_sign": y0_sign, **st})

    hyp_df = pd.DataFrame(hyp_rows).sort_values("score").reset_index(drop=True)
    print("\n[HYP] ranking phase1:")
    print(hyp_df[["order","conv_rot","image2ground","y0_sign","med","rmse","p95","score"]].to_string(index=False))

    # gating + dedup
    best_score, best_rmse = float(hyp_df.loc[0, "score"]), float(hyp_df.loc[0, "rmse"])
    kept_df = hyp_df[np.isfinite(hyp_df["rmse"]) & np.isfinite(hyp_df["score"])].copy()
    kept_df = kept_df[(kept_df["rmse"] <= 150.0) & (kept_df["score"] <= 1.8 * best_score) & (kept_df["rmse"] <= 1.6 * best_rmse)]
    kept_df = kept_df.sort_values("score").drop_duplicates(subset=["order","conv_rot","image2ground"], keep="first").reset_index(drop=True)
    if kept_df.empty:
        kept_df = hyp_df.head(1).copy()

    print("\n[HYP] kept:")
    print(kept_df[["order","conv_rot","image2ground","y0_sign","med","rmse","p95","score"]].to_string(index=False))

    final_rows, final_logs = [], []
    best_ref_score = None

    if fixed_cam is None:
        # phase2 rapide: dc/dl/k1 only
        fit = {
            "coarse_r": {"dc": args.coarse_r_dc, "dl": args.coarse_r_dl, "k1": args.coarse_r_k1, "domega": 0.0, "dphi": 0.0},
            "coarse_s": {"dc": 3, "dl": 3, "k1": 3, "domega": 1, "dphi": 1},
            "ref1_r": {"dc": args.ref1_r_dc, "dl": args.ref1_r_dl, "k1": args.ref1_r_k1, "domega": 0.0, "dphi": 0.0},
            "ref1_s": {"dc": 3, "dl": 3, "k1": 3, "domega": 1, "dphi": 1},
            "ref2_r": {"dc": args.ref2_r_dc, "dl": args.ref2_r_dl, "k1": args.ref2_r_k1, "domega": 0.0, "dphi": 0.0},
            "ref2_s": {"dc": 3, "dl": 3, "k1": 3, "domega": 1, "dphi": 1},
        }

        for i, h in kept_df.iterrows():
            print(f"\n[PHASE2] hypothesis {i+1}/{len(kept_df)} => {h['order']} {h['conv_rot']} i2g={h['image2ground']} y0={int(h['y0_sign'])}")
            samples = build_samples(df, images_dir, obs_dir, int(h["y0_sign"]), args.fit_stride, args.fit_max_images)
            cfg = {"order": str(h["order"]), "conv_rot": str(h["conv_rot"]), "image2ground": bool(h["image2ground"]), "z_positive": (not args.z_negative)}

            best, logs, stage_bests = optimize_phase2(samples, tp3d, cfg, fit, use_input_ppa=args.use_input_ppa)

            c = stage_bests["coarse"]
            if best_ref_score is not None:
                if (c["rmse"] > args.phase2_abs_abort_rmse) or (c["score"] > args.phase2_rel_abort * best_ref_score):
                    print(f"[PHASE2] ABORT hyp {i+1}: coarse too bad (rmse={c['rmse']:.1f}, score={c['score']:.1f})")
                    continue

            if best_ref_score is None or best["score"] < best_ref_score:
                best_ref_score = best["score"]

            logs = logs.copy()
            logs["order"] = h["order"]; logs["conv_rot"] = h["conv_rot"]; logs["image2ground"] = h["image2ground"]; logs["y0_sign"] = h["y0_sign"]
            final_logs.append(logs)

            final_rows.append({
                "order": h["order"], "conv_rot": h["conv_rot"], "image2ground": h["image2ground"], "y0_sign": int(h["y0_sign"]),
                "dc": best["dc"], "dl": best["dl"], "k1": best["k1"], "domega": best["domega"], "dphi": best["dphi"],
                "med": best["med"], "rmse": best["rmse"], "p95": best["p95"], "score": best["score"]
            })

        final_df = pd.DataFrame(final_rows).sort_values("score").reset_index(drop=True)
        if final_df.empty:
            raise RuntimeError("Aucune hypothèse valide après phase2.")

        best_row = final_df.iloc[0].to_dict()

        # refine angulaire final
        samples_best = build_samples(df, images_dir, obs_dir, int(best_row["y0_sign"]), args.fit_stride, args.fit_max_images)
        cfg_best = {
            "order": str(best_row["order"]),
            "conv_rot": str(best_row["conv_rot"]),
            "image2ground": bool(best_row["image2ground"]),
            "z_positive": (not args.z_negative)
        }

        ang_best, ang_logs = optimize_domega_dphi_final(
            samples=samples_best, tp3d=tp3d, cfg=cfg_best,
            fixed_params={"dc": float(best_row["dc"]), "dl": float(best_row["dl"]), "k1": float(best_row["k1"])},
            use_input_ppa=args.use_input_ppa
        )

        ang_top3 = ang_logs.sort_values("score").head(3).copy()
        print("\n[ANG] top3:")
        print(ang_top3[["stage","domega","dphi","med","rmse","p95","score"]].to_string(index=False))

        best_row["domega"] = float(ang_best["domega"])
        best_row["dphi"] = float(ang_best["dphi"])
        best_row["med"] = float(ang_best["med"])
        best_row["rmse"] = float(ang_best["rmse"])
        best_row["p95"] = float(ang_best["p95"])
        best_row["score"] = float(ang_best["score"])

        print(f"[FINAL+ANG] dc={best_row['dc']:.3f} dl={best_row['dl']:.3f} k1={best_row['k1']:.6f} dω={best_row['domega']:.4f} dφ={best_row['dphi']:.4f} rmse={best_row['rmse']:.2f}")
    else:
        # Intrinsics figées via camera_txt: pas d'estimation dc/dl/k1
        h = kept_df.iloc[0]
        best_row = {
            "order": h["order"], "conv_rot": h["conv_rot"], "image2ground": bool(h["image2ground"]), "y0_sign": int(h["y0_sign"]),
            "dc": 0.0, "dl": 0.0, "k1": float(fixed_cam["k1"]),
            "domega": 0.0, "dphi": 0.0,
            "med": float(h["med"]), "rmse": float(h["rmse"]), "p95": float(h["p95"]), "score": float(h["score"])
        }
        final_df = pd.DataFrame([best_row])
        print("[INFO] camera_txt fourni => optimisation intrinsics (dc/dl/k1) désactivée.")

    return {
        "args": args,
        "df": df,
        "tp3d": tp3d,
        "images_dir": images_dir,
        "obs_dir": obs_dir,
        "out_dir": out_dir,
        "hyp_df": hyp_df,
        "final_df": final_df,
        "best_row": best_row,
        "final_logs": final_logs,
        "fixed_cam": fixed_cam
    }


# ========================= 3) export best hypothesis =========================

def export_best_hyp(result):
    args = result["args"]
    df = result["df"]
    tp3d = result["tp3d"]
    images_dir = result["images_dir"]
    obs_dir = result["obs_dir"]
    out_dir = result["out_dir"]
    hyp_df = result["hyp_df"]
    final_df = result["final_df"]
    best_row = result["best_row"]
    final_logs = result["final_logs"]
    fixed_cam = result.get("fixed_cam", None)

    print("\n[FINAL]")
    print(pd.DataFrame([best_row]).to_string(index=False))

    if final_logs:
        pd.concat(final_logs, ignore_index=True).to_csv(Path(args.fit_report_csv) if args.fit_report_csv else out_dir/"fit_phase2.csv", index=False)
    hyp_df.to_csv(Path(args.hyp_report_csv) if args.hyp_report_csv else out_dir/"hyp_phase1.csv", index=False)

    cfg_best = {
        "order": str(best_row["order"]),
        "conv_rot": str(best_row["conv_rot"]),
        "image2ground": bool(best_row["image2ground"]),
        "z_positive": (not args.z_negative)
    }

    report_rows, n_written = [], 0
    for _, row in df.iterrows():
        stem = Path(str(row["label"])).stem
        jpg = images_dir / f"{stem}.jpg"
        if not jpg.exists():
            continue
        with Image.open(jpg) as im:
            w, h = im.size

        if fixed_cam is not None:
            ppa_c = float(fixed_cam["cx"])
            ppa_l = float(fixed_cam["cy"])
            k1_val = float(fixed_cam["k1"])
            foc_override = 0.5 * (float(fixed_cam["fx"]) + float(fixed_cam["fy"]))  # CON n'a qu'une focale
        else:
            base_cx = float(row["x0"]) if args.use_input_ppa else w/2.0
            base_cy = float(best_row["y0_sign"]) * float(row["y0"]) if args.use_input_ppa else h/2.0
            ppa_c = base_cx + float(best_row["dc"])
            ppa_l = base_cy + float(best_row["dl"])
            k1_val = float(best_row["k1"])
            foc_override = None

        tree, img_name = build_con(
            row, ppa_c, ppa_l, w, h,
            float(best_row["domega"]), float(best_row["dphi"]),
            cfg_best["order"], cfg_best["conv_rot"], cfg_best["image2ground"],
            focale_override=foc_override
        )
        indent(tree.getroot())
        tree.write(out_dir / f"{img_name}.CON", encoding="utf-8", xml_declaration=True)
        n_written += 1

        obsf = obs_dir / f"{stem}_obs.txt"
        if obsf.exists():
            obs_rows = parse_obs_file(obsf)
            if obs_rows:
                om = float(row["omega[deg]"]) + float(best_row["domega"])
                ph = float(row["phi[deg]"]) + float(best_row["dphi"])
                ka = float(row["kappa[deg]"])
                R = apply_conv(opk_to_R(om, ph, ka, order=cfg_best["order"]), cfg_best["conv_rot"])

                errs, behind = [], 0
                for pid, u_gt, v_gt in obs_rows:
                    xyz = tp3d.get(pid)
                    if xyz is None:
                        continue
                    proj = project_point(
                        xyz[0], xyz[1], xyz[2],
                        float(row["X0"]), float(row["Y0"]), float(row["Z0"]),
                        R, ppa_c, ppa_l, float(row["c"]) if foc_override is None else float(foc_override), w, h,
                        image2ground=cfg_best["image2ground"], z_positive=cfg_best["z_positive"],
                        k1=k1_val
                    )
                    if proj is None:
                        behind += 1
                        continue
                    u, v = proj
                    errs.append(math.hypot(u-u_gt, v-v_gt))

                st = robust_stats(errs)
                report_rows.append({"image": stem, "behind": behind, **st})

    report_path = Path(args.report_csv) if args.report_csv else out_dir/"con_validation_report.csv"
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    print(f"[INFO] validation report: {report_path}")
    print(f"OK: {n_written} CON écrits")


# ========================= main =========================

def main():
    args = parse_args()
    result = run_optimisation(args)
    export_best_hyp(result)


if __name__ == "__main__":
    main()

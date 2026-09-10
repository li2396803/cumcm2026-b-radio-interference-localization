"""问题 2: 第二检测点选择策略的定量优化

模型(参考坐标: S1 为原点, 第一次示向度 theta1 沿 +x 轴):
  干扰源先验 = 楔形 W1(theta1 ± 1°) ∩ B(S1,1500) 内均匀  (在 S1 处已测到信号 => rho1 <= 1500)
  选择第二点 S2 后:
    命中(direction, 概率 P=Pr[R>=rho2], R~U(1000,1500)) : 定位区域 = W1 ∩ W2 ∩ B(S1,1500)
    无信号(no_signal)                                     : 排除 B(S2,1000) 后剩余区域
  目标: 极小化 J(S2)=E[log D], D 为定位区域直径(保守地取 no_signal 分支的跨度估计)
输出: results/p2_grid.json, results/p2_summary.json
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))
from geom import (  # noqa: E402
    ang_deg, diameter_rotating_calipers_pair, localization_polygon, mec, norm,
    polygon_area, sample_in_polygon, unit, convex_hull,
)

EPS = 1.0
R_MAX = 1500.0
R_MIN = 1000.0
N_SRC = 240
D_GRID = np.arange(200.0, 1701.0, 100.0)
A_GRID = np.arange(0.0, 180.1, 7.5)


def make_samples(rng, n=N_SRC):
    """均匀采样 rho1 (密度 ∝ rho) 与角度误差 eps1。"""
    rho = R_MAX * np.sqrt(rng.random(n))
    eps = rng.uniform(-EPS, EPS, n)
    return rho, eps


def source_xy(rho, eps, S1=(0.0, 0.0), theta1=0.0):
    out = np.zeros((len(rho), 2))
    for i in range(len(rho)):
        d = unit(theta1 + eps[i])
        out[i] = np.asarray(S1, float) + rho[i] * d
    return out


def region_exact(S1, t1, S2, t2, arena=R_MAX):
    poly, _ = localization_polygon([(S1, t1, EPS), (S2, t2, EPS)], arena=arena)
    if len(poly) < 2:
        return 0.0, 0.0, 0.0
    D, _ = diameter_rotating_calipers_pair(poly)
    return D, polygon_area(poly), len(poly)


def nosignal_span(S1, t1, S2, pts):
    """W1∩B(S1,1500) 去掉 B(S2,1000) 后剩余区域的跨度(用主方向极差估计)。"""
    keep = np.linalg.norm(pts - np.asarray(S2, float), axis=1) > R_MIN
    p = pts[keep]
    if len(p) < 5:
        return 0.0
    q = p - p.mean(axis=0)
    cov = q.T @ q / len(q)
    w, v = np.linalg.eigh(cov)
    u = v[:, -1]
    t = q @ u
    return float(t.max() - t.min())


def main():
    res_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(res_dir, exist_ok=True)
    rng = np.random.default_rng(2026)
    S1, t1 = np.array([0.0, 0.0]), 0.0
    rho, eps = make_samples(rng)
    src = source_xy(rho, eps)
    # W1∩B(S1,1500) 采样点(用于 no_signal 分支)
    poly1, _ = localization_polygon([(S1, t1, EPS)], arena=R_MAX)
    wpts = sample_in_polygon(poly1, 4000, np.random.default_rng(11))

    noise = np.random.default_rng(7)
    u_det = noise.random(400000).reshape(-1)          # 预生成随机数
    u_err = noise.uniform(-EPS, EPS, 400000)
    k = 0
    grid = []
    for d in D_GRID:
        for a in A_GRID:
            S2 = S1 + d * unit(t1 + a)
            rho2 = np.linalg.norm(src - S2, axis=1)
            pdet = np.clip((R_MAX - rho2) / (R_MAX - R_MIN), 0.0, 1.0)
            k0 = k
            det = u_det[k0:k0 + len(src)] < pdet
            k += len(src)
            logs = np.empty(len(src))
            dd = math.log(max(nosignal_span(S1, t1, S2, wpts), 1.0))
            det_list = []
            for i in range(len(src)):
                if det[i]:
                    t2 = ang_deg(src[i] - S2) + u_err[k0 + i]
                    D, _, _ = region_exact(S1, t1, S2, t2 % 360.0)
                    D = max(D, 1.0)
                    det_list.append(D)
                    logs[i] = math.log(D)
                else:
                    logs[i] = dd
            grid.append({
                "d": float(d), "alpha": float(a), "S2": [float(S2[0]), float(S2[1])],
                "p_detect": float(pdet.mean()),
                "E_diam_geom": float(math.exp(logs.mean())),
                "E_log_diam": float(logs.mean()),
                "E_diam_detected": float(np.mean(det_list)) if det_list else None,
                "n_detected": len(det_list),
                "span_no_signal": float(math.exp(dd)),
            })
        print(f"  d={d:.0f} done", flush=True)

    best = min(grid, key=lambda g: g["E_log_diam"])
    thr = best["E_log_diam"] + math.log(1.05)
    near = [g for g in grid if g["E_log_diam"] <= thr]
    det_ok = [g for g in grid if g["E_diam_detected"] is not None]
    front = []
    for g in sorted(det_ok, key=lambda g: -g["E_diam_detected"]):
        if not front or g["p_detect"] > front[-1]["p_detect"] + 1e-9:
            front.append(g)

    summary = {
        "setting": {"S1": S1.tolist(), "theta1": t1, "source_prior":
                    "W1(theta1±1°) ∩ B(S1,1500) 内均匀; R~U(1000,1500)",
                    "n_src": N_SRC, "d_grid": D_GRID.tolist(), "alpha_grid": A_GRID.tolist()},
        "best": best, "n_near_optimal": len(near), "near_optimal": near,
        "pareto": front,
    }
    with open(os.path.join(res_dir, "p2_grid.json"), "w") as f:
        json.dump(grid, f, ensure_ascii=False, indent=2)
    with open(os.path.join(res_dir, "p2_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("best:", json.dumps(best, ensure_ascii=False))
    print(f"near-optimal set ({len(near)} grid points):")
    for g in near:
        print(f"  d={g['d']:6.0f} a={g['alpha']:6.1f} p_det={g['p_detect']:.3f} "
              f"E[D]={g['E_diam_geom']:8.1f} E[D|det]="
              f"{g['E_diam_detected'] if g['E_diam_detected'] is None else round(g['E_diam_detected'],1)}")
    print("pareto:")
    for g in front:
        print(f"  p_det={g['p_detect']:.3f} E[D|det]={g['E_diam_detected']:.1f} "
              f"d={g['d']:.0f} a={g['alpha']:.1f}")


if __name__ == "__main__":
    main()

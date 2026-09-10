"""问题 1 数值研究

(1) 三种直径算法互校 (枚举 / 旋转卡壳 / 支撑函数扫描)
(2) 定位区域直径、最小包围圆(MEC)统计规律
(3) "以定位区域直径为直径的圆是否覆盖定位区域" 判定 + 最坏反例
(4) 结构判据验证: 交会区域有界 <=> 各示向度误差弧无公共方向
输出: results/p1_summary.json
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))
from geom import (  # noqa: E402
    ang_deg, diameter_bruteforce, diameter_disk_covers, diameter_rotating_calipers,
    diameter_rotating_calipers_pair, diameter_support_scan, is_bounded,
    localization_polygon, mec, point_in_polygon, polygon_area, wedge_halfplanes,
)

R_ARENA = 1800.0
EPS_DEG = 1.0
RES = os.path.join(os.path.dirname(__file__), "results")
GRID = np.arange(0.0, 360.0, 0.25)


def rand_point_in_disk(rng, R):
    r = R * math.sqrt(rng.random())
    a = rng.random() * 2 * math.pi
    return np.array([r * math.cos(a), r * math.sin(a)])


def region_of(pts, thetas, arena=R_ARENA):
    obs = [(pts[i], thetas[i], EPS_DEG) for i in range(len(pts))]
    return localization_polygon(obs, arena=arena)


def arc_common(thetas):
    """各误差弧是否共享一个公共方向 (矢量化)。"""
    diff = np.abs((GRID[:, None] - np.array(thetas)[None, :] + 180.0) % 360.0 - 180.0)
    return bool((diff <= EPS_DEG + 1e-9).all(axis=1).any())


def main():
    os.makedirs(RES, exist_ok=True)
    rng = np.random.default_rng(20260910)
    out = {}

    # ---------- (1) 算法互校 ----------
    d_cp, d_sup, d_bf = [], [], []
    for _ in range(3000):
        n = int(rng.integers(2, 6))
        G = rand_point_in_disk(rng, R_ARENA)
        pts = [rand_point_in_disk(rng, R_ARENA) for _ in range(n)]
        ths = [(ang_deg(G - p) + rng.uniform(-1, 1)) % 360 for p in pts]
        poly, _ = region_of(pts, ths)
        if len(poly) < 2:
            continue
        d0 = diameter_bruteforce(poly) if len(poly) <= 24 else diameter_rotating_calipers(poly)
        d1 = diameter_rotating_calipers(poly)
        d2 = diameter_support_scan(poly)
        if d0 > 1e-6:
            d_cp.append(abs(d1 - d0) / d0)
            d_sup.append(abs(d2 - d0) / d0)
            d_bf.append(d0)
    out["algo_check"] = {
        "n_cases": len(d_cp),
        "max_rel_err_calipers": float(np.max(d_cp)),
        "max_rel_err_support_scan": float(np.max(d_sup)),
        "mean_diameter_m": float(np.mean(d_bf)),
    }

    # ---------- (4) 无界性判据 ----------
    agree = []
    for _ in range(4000):
        n = 2 if rng.random() < 0.5 else int(rng.integers(3, 6))
        G = rand_point_in_disk(rng, R_ARENA)
        pts = [rand_point_in_disk(rng, R_ARENA) for _ in range(n)]
        ths = [(ang_deg(G - p) + rng.uniform(-1, 1)) % 360 for p in pts]
        hps = []
        for i in range(n):
            hps.extend(wedge_halfplanes(pts[i], ths[i], EPS_DEG))
        agree.append(is_bounded(hps) == (not arc_common(ths)))
    out["boundedness_rule_check"] = {"n": len(agree), "agreement_rate": float(np.mean(agree))}

    # 2 点情形解析判据: |dtheta| > 2 eps  <=>  有界
    hit, tot = 0, 0
    for _ in range(4000):
        G = rand_point_in_disk(rng, R_ARENA)
        pts = [rand_point_in_disk(rng, R_ARENA) for _ in range(2)]
        ths = [(ang_deg(G - p) + rng.uniform(-1, 1)) % 360 for p in pts]
        hps = []
        for i in range(2):
            hps.extend(wedge_halfplanes(pts[i], ths[i], EPS_DEG))
        dth = abs(((ths[0] - ths[1] + 180) % 360) - 180)
        tot += 1
        hit += int(is_bounded(hps) == (dth > 2 * EPS_DEG))
    out["two_point_rule_check"] = {"n": tot, "agreement_rate": hit / tot}

    # ---------- (2)(3) 覆盖性质统计 ----------
    # 场景 A: 检测点在目标区域内随机;  场景 B: 检测点环绕干扰源(实际"良好交会"配置)
    for scen in ("A", "B"):
        for npts, trials in ((2, 3000), (3, 3000), (4, 2000), (5, 2000)):
            _run_cover_block(out, rng, scen, npts, trials)

    worst_all = max((out[f"cover_{sc}_{k}"]["worst_case"]
                     for sc in ("A", "B") for k in (2, 3, 4, 5)), key=lambda w: w["ratio"])
    out["sharpest_counterexample"] = worst_all
    out["jung_bound_2_over_sqrt3"] = 2.0 / math.sqrt(3.0)
    out["observed_max_ratio"] = max(out[f"cover_{sc}_{k}"]["max_ratio_2R_over_D"]
                                    for sc in ("A", "B") for k in (2, 3, 4, 5))

    w = worst_all
    poly, _ = region_of([np.array(p) for p in w["points"]], w["thetas"])
    out["counterexample_verify"] = {
        "target_inside_region": bool(point_in_polygon(np.array(w["G"]), poly)),
        "diameter_disk_covers": bool(diameter_disk_covers(
            poly, (np.array(w["diam_pair"][0]), np.array(w["diam_pair"][1])))),
        "mec_radius": w["mec_r"], "D_half": w["D"] / 2, "ratio": w["ratio"],
    }

    with open(os.path.join(RES, "p1_summary.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("sharpest_counterexample",)}, ensure_ascii=False, indent=2))


def _run_cover_block(out, rng, scen, npts, trials):
    ratios, covered, diams, areas = [], 0, [], []
    worst = None
    for _ in range(trials):
        G = rand_point_in_disk(rng, R_ARENA)
        if scen == "A":
            pts = [rand_point_in_disk(rng, R_ARENA) for _ in range(npts)]
        else:
            base = rng.uniform(0, 2 * math.pi)
            pts = []
            for i in range(npts):
                a = base + 2 * math.pi * i / npts + rng.normal(0, 0.25)
                rr = rng.uniform(600, 1500)
                pts.append(G + rr * np.array([math.cos(a), math.sin(a)]))
        ths = [(ang_deg(G - p) + rng.uniform(-1, 1)) % 360 for p in pts]
        poly, bounded = region_of(pts, ths)
        if len(poly) < 3:
            continue
        D, pair = diameter_rotating_calipers_pair(poly)
        if D < 1e-9:
            continue
        c, r = mec(poly)
        ratio = 2 * r / D
        ratios.append(ratio)
        diams.append(D)
        areas.append(polygon_area(poly))
        ok = bool(r <= D / 2 * (1 + 1e-9) + 1e-6)
        covered += int(ok)
        if worst is None or ratio > worst["ratio"]:
            worst = {"ratio": float(ratio), "points": [p.tolist() for p in pts],
                     "thetas": [float(t) for t in ths], "G": [float(G[0]), float(G[1])],
                     "D": float(D), "mec_r": float(r), "mec_c": c.tolist(),
                     "diam_pair": [pair[0].tolist(), pair[1].tolist()],
                     "covered_by_diameter_disk": ok, "bounded": bool(bounded),
                     "area_m2": float(polygon_area(poly)), "n_vertices": len(poly)}
    m = len(ratios)
    out[f"cover_{scen}_{npts}"] = {
        "trials_used": m,
        "frac_covered_by_diameter_disk": covered / m,
        "mean_ratio_2R_over_D": float(np.mean(ratios)),
        "p95_ratio_2R_over_D": float(np.percentile(ratios, 95)),
        "max_ratio_2R_over_D": float(np.max(ratios)),
        "mean_diameter_m": float(np.mean(diams)),
        "median_diameter_m": float(np.median(diams)),
        "mean_area_m2": float(np.mean(areas)),
        "worst_case": worst,
    }


if __name__ == "__main__":
    main()

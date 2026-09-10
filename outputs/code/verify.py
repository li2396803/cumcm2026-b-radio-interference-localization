"""检验脚本: 模拟器一致性 / 几何算法互校 / 策略不变量 / 对抗性算例 / 复现性。

输出: results/verify.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))
from geom import (  # noqa: E402
    ang_deg, diameter_bruteforce, diameter_rotating_calipers, localization_polygon,
    mec_bruteforce, mec_minidisk,
)
from sim import R_ARENA, Simulator, Source, generate_case  # noqa: E402
from strategy import Params, run_trial  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "results")


def test_simulator_timing():
    """复现附件1 第3节示例: 虚拟时刻应为 105/111/194/199。"""
    srcs = [Source(3, np.array([1000.0, 2000.0]), 1500.0, "omni")]
    sim = Simulator(srcs)
    sim.measure(np.array([300.0, 400.0]), 1)
    t1 = sim.t
    sim.measure(np.array([300.0, 400.0]), 2)
    t2 = sim.t
    sim.clear(np.array([300.0, 0.0]), 3)
    t3 = sim.t
    sim.measure(np.array([300.0, 0.0]), 2)
    t4 = sim.t
    return {"observed": [t1, t2, t3, t4], "expected": [105, 111, 194, 199],
            "pass": [t1, t2, t3, t4] == [105.0, 111.0, 194.0, 199.0]}


def test_geometry_algorithms():
    rng = np.random.default_rng(1)
    err_cp, err_mec, n = [], [], 0
    for _ in range(300):
        k = int(rng.integers(2, 5))
        G = np.array([rng.uniform(-1500, 1500), rng.uniform(-1500, 1500)])
        pts = [np.array([rng.uniform(-1800, 1800), rng.uniform(-1800, 1800)]) for _ in range(k)]
        th = [(ang_deg(G - p) + rng.uniform(-1, 1)) % 360 for p in pts]
        poly, _ = localization_polygon([(pts[i], th[i], 1.0) for i in range(k)], arena=R_ARENA)
        if len(poly) < 2:
            continue
        n += 1
        d0 = diameter_bruteforce(poly) if len(poly) <= 24 else diameter_rotating_calipers(poly)
        if d0 > 1e-9:
            err_cp.append(abs(diameter_rotating_calipers(poly) - d0) / d0)
        if len(poly) <= 12:
            _, r1 = mec_minidisk(poly)
            _, r2 = mec_bruteforce(poly)
            err_mec.append(abs(r1 - r2))
    return {"n_cases": n,
            "max_rel_err_calipers_vs_bruteforce": float(np.max(err_cp)),
            "max_abs_err_mec_vs_bruteforce_m": float(np.max(err_mec)),
            "pass": float(np.max(err_cp)) < 1e-6 and float(np.max(err_mec)) < 1e-6}


def test_strategy_invariants(seed0, n, problem, cover_pts):
    rows = []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        srcs = generate_case(rng, problem)
        res = run_trial(srcs, cover_pts, Params(name="A"), problem)
        st = res["stats"]
        tot = st["t_move"] + st["t_switch"] + st["t_measure"] + st["t_clear"]
        rows.append({"seed": seed0 + i,
                     "time_gap_s": abs(tot - st["total_time_s"]),
                     "within_virtual_limit": st["total_time_s"] <= 360000.0,
                     "cleared_matches_truth": st["n_cleared"] == sum(1 for s in srcs if s.cleared),
                     "resolution_consistent": (len(res["cleared"]) + len(res["proven_empty"])
                                               + len(res["unresolved"])) <= 20,
                     "ratio": st["ratio"]})
    return {"n_cases": n,
            "max_time_breakdown_gap_s": float(max(r["time_gap_s"] for r in rows)),
            "all_within_virtual_limit": bool(all(r["within_virtual_limit"] for r in rows)),
            "all_cleared_flags_consistent": bool(all(r["cleared_matches_truth"] for r in rows)),
            "accounts_consistent": bool(all(r["resolution_consistent"] for r in rows)),
            "mean_ratio": float(np.mean([r["ratio"] for r in rows])),
            "n_full_clear": int(sum(1 for r in rows if r["ratio"] >= 1 - 1e-9))}


def test_adversarial(cover_pts, problem, n=25, seed0=424242):
    """对抗性算例: 所有源取最小有效接收半径 1000m, 且位置贴外圈。"""
    rows = []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        k = int(rng.integers(10, 17))
        chans = rng.choice(np.arange(1, 21), size=k, replace=False)
        srcs = []
        for ch in chans:
            a = rng.random() * 2 * np.pi
            rr = R_ARENA * (0.85 + 0.15 * rng.random())
            pos = np.array([rr * np.cos(a), rr * np.sin(a)])
            if problem == 3:
                srcs.append(Source(int(ch), pos, 1000.0, "omni"))
            else:
                kind = "dir" if rng.random() < 0.5 else "omni"
                srcs.append(Source(int(ch), pos, 1000.0, kind,
                                   None if kind == "omni" else float(rng.random() * 360.0)))
        res = run_trial(srcs, cover_pts, Params(name="A"), problem)
        st = res["stats"]
        rows.append({"seed": seed0 + i, "n": st["n_sources"], "cleared": st["n_cleared"],
                     "ratio": st["ratio"], "avg": st["avg_time_s"]})
    tot = sum(r["n"] for r in rows)
    return {"n_cases": n, "reception_radius_m": 1000.0,
            "ratio_overall": sum(r["cleared"] for r in rows) / tot,
            "n_full_clear": int(sum(1 for r in rows if r["ratio"] >= 1 - 1e-9)),
            "mean_avg_time_s": float(np.mean([r["avg"] for r in rows]))}


def test_reproducibility(cover_pts, problem, seeds=(111, 222, 333)):
    out = []
    for sd in seeds:
        res = []
        for _ in range(2):
            rng = np.random.default_rng(sd)
            srcs = generate_case(rng, problem)
            r = run_trial(srcs, cover_pts, Params(name="A"), problem)
            res.append((r["stats"]["total_time_s"], r["n_measure"], r["n_clear"]))
        out.append({"seed": sd, "run1": res[0], "run2": res[1], "identical": res[0] == res[1]})
    return {"all_identical": bool(all(o["identical"] for o in out)), "detail": out}


def main():
    cov = json.load(open(os.path.join(RES, "cover_points.json")))
    p3 = cov["problem3"]["points_ordered"]
    p4 = {round(t["spacing"]): t["points_ordered"] for t in cov["problem4_tradeoff"]}[1000]
    out = {
        "simulator_timing_example": test_simulator_timing(),
        "geometry_algorithms": test_geometry_algorithms(),
        "invariants_p3": test_strategy_invariants(500000, 40, 3, p3),
        "invariants_p4": test_strategy_invariants(500000, 40, 4, p4),
        "adversarial_p3": test_adversarial(p3, 3),
        "adversarial_p4": test_adversarial(p4, 4),
        "reproducibility_p3": test_reproducibility(p3, 3),
        "reproducibility_p4": test_reproducibility(p4, 4),
    }
    with open(os.path.join(RES, "verify.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    for k, v in out.items():
        vv = {kk: val for kk, val in v.items() if kk != "detail"} if isinstance(v, dict) else v
        print(k, json.dumps(vv, ensure_ascii=False)[:500], flush=True)


if __name__ == "__main__":
    main()

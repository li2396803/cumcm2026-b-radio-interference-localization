"""v2 策略的检验: 不变量 / 对抗算例 / 复现性 / 性能对比(与 v1 同种子)。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))
sys.path.insert(0, os.path.dirname(__file__))
from sim import R_ARENA, Source, generate_case  # noqa: E402
from strategy2 import Params2, run_trial_v2  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "results")


def load():
    cov = json.load(open(os.path.join(RES, "cover_points.json")))
    s3 = cov["problem3"]["points_ordered"]
    p4 = {round(t["spacing"]): t["points_ordered"] for t in cov["problem4_tradeoff"]}[1000]
    return {"level1": s3, "escalation": []}, {"level1": p4, "escalation": []}


def invariants(plan, problem, n=40, seed0=500000):
    rows = []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        srcs = generate_case(rng, problem)
        r = run_trial_v2(srcs, plan, Params2(), problem)
        st = r["stats"]
        tot = st["t_move"] + st["t_switch"] + st["t_measure"] + st["t_clear"]
        rows.append({"gap": abs(tot - st["total_time_s"]),
                     "limit_ok": st["total_time_s"] <= 360000.0,
                     "flag_ok": st["n_cleared"] == sum(1 for s in srcs if s.cleared),
                     "ratio": st["ratio"]})
    return {"n": n,
            "max_time_gap_s": float(max(r["gap"] for r in rows)),
            "all_within_limit": bool(all(r["limit_ok"] for r in rows)),
            "all_flags_consistent": bool(all(r["flag_ok"] for r in rows)),
            "mean_ratio": float(np.mean([r["ratio"] for r in rows])),
            "n_full_clear": int(sum(1 for r in rows if r["ratio"] >= 1 - 1e-9))}


def adversarial(plan, problem, n=25, seed0=424242):
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
        r = run_trial_v2(srcs, plan, Params2(), problem)
        rows.append((r["stats"]["n_sources"], r["stats"]["n_cleared"],
                     r["stats"]["avg_time_s"] or 0))
    tot = sum(a for a, _, _ in rows)
    cl = sum(b for _, b, _ in rows)
    return {"n_cases": n, "ratio": cl / tot,
            "n_full_clear": int(sum(1 for a, b, _ in rows if a == b)),
            "mean_avg_time_s": float(np.mean([c for _, _, c in rows]))}


def reproducibility(plan, problem, seeds=(111, 222, 333)):
    out = []
    for sd in seeds:
        res = []
        for _ in range(2):
            rng = np.random.default_rng(sd)
            srcs = generate_case(rng, problem)
            r = run_trial_v2(srcs, plan, Params2(), problem)
            res.append((r["stats"]["total_time_s"], r["stats"]["n_requests"]))
        out.append(res[0] == res[1])
    return {"all_identical": bool(all(out)), "detail": out}


def main():
    p3, p4 = load()
    res = {
        "invariants_p3": invariants(p3, 3),
        "invariants_p4": invariants(p4, 4),
        "adversarial_p3": adversarial(p3, 3),
        "adversarial_p4": adversarial(p4, 4),
        "reproducibility_p3": reproducibility(p3, 3),
        "reproducibility_p4": reproducibility(p4, 4),
    }
    with open(os.path.join(RES, "verify2.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    for k, v in res.items():
        vv = {kk: val for kk, val in v.items() if kk != "detail"}
        print(k, json.dumps(vv, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

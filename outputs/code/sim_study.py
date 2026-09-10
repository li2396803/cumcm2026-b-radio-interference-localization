"""问题 3 / 问题 4 策略演练与统计检验。

策略对比:
  A  adaptive : 侦察-清除插序 + 交会追踪 + 清除扫描 (本文策略)
  B  sequential: 先完成全部侦察, 再依次清除 (默认几何, 无插序)
  C  nosweep  : 插序 + 只靠交会追踪(不使用清除扫描)
问题4 额外比较三套侦察点集(7 / 13 / 19 点, 环绕条件漏检率不同)。
输出: results/sim_p3_*.json, results/sim_p4_*.json, results/sim_summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))
sys.path.insert(0, os.path.dirname(__file__))
from sim import R_ARENA, generate_case  # noqa: E402
from strategy import Params, run_trial  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "results")


def load_cover():
    with open(os.path.join(RES, "cover_points.json")) as f:
        d = json.load(f)
    return d


def aggregate(rows):
    st = [r["stats"] for r in rows]
    ratios = np.array([s["ratio"] for s in st])
    avg = np.array([s["avg_time_s"] if s["avg_time_s"] else np.nan for s in st])
    total = np.array([s["total_time_s"] for s in st])
    nsrc = np.array([s["n_sources"] for s in st])
    ncl = np.array([s["n_cleared"] for s in st])
    return {
        "n_trials": len(rows),
        "mean_ratio": float(np.nanmean(ratios)),
        "min_ratio": float(np.nanmin(ratios)),
        "frac_full_clear": float(np.mean(ratios >= 1.0 - 1e-9)),
        "mean_avg_time_s": float(np.nanmean(avg)),
        "median_avg_time_s": float(np.nanmedian(avg)),
        "std_avg_time_s": float(np.nanstd(avg)),
        "p10_avg_time_s": float(np.nanpercentile(avg, 10)),
        "p90_avg_time_s": float(np.nanpercentile(avg, 90)),
        "mean_total_time_s": float(np.mean(total)),
        "mean_n_sources": float(np.mean(nsrc)),
        "mean_n_cleared": float(np.mean(ncl)),
        "total_missed": int(np.sum(nsrc - ncl)),
        "mean_requests2": float(np.mean([r["stats"]["n_requests"] for r in rows])),
        "mean_move_m": float(np.mean([s["move_m"] for s in st])),
        "time_breakdown": {
            "move": float(np.mean([s["t_move"] for s in st])),
            "switch": float(np.mean([s["t_switch"] for s in st])),
            "measure": float(np.mean([s["t_measure"] for s in st])),
            "clear": float(np.mean([s["t_clear"] for s in st])),
        },
        "mean_home_steps": float(np.mean([r["n_home_steps"] for r in rows])),
        "mean_clear_ops": float(np.mean([r["n_clear"] for r in rows])),
        "mean_anomaly": float(np.mean([r["anomaly"] for r in rows])),
    }


def run_block(problem, params, cover_pts, n_trials, seed0, tag):
    rows = []
    t0 = time.time()
    for i in range(n_trials):
        rng = np.random.default_rng(seed0 + i)
        srcs = generate_case(rng, problem)
        res = run_trial(srcs, cover_pts, params, problem)
        res["case_seed"] = seed0 + i
        rows.append(res)
    agg = aggregate(rows)
    agg.update({"tag": tag, "problem": problem, "params": params.__dict__,
                "wall_s": time.time() - t0})
    # 保存逐局结果(只留关键字段, 控制体积)
    with open(os.path.join(RES, f"sim_{tag}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({"seed": r["case_seed"], "stats": r["stats"],
                                "cleared": r["cleared"], "proven_empty": r["proven_empty"],
                                "unresolved": r["unresolved"],
                                "n_home_steps": r["n_home_steps"],
                                "anomaly": r["anomaly"],
                                "sources": r["sources"]}, ensure_ascii=False) + "\n")
    return agg, rows


def main():
    os.makedirs(RES, exist_ok=True)
    cover = load_cover()
    p3_pts = cover["problem3"]["points_ordered"]
    trade = {round(t["spacing"]): t for t in cover["problem4_tradeoff"]}
    p4_small = trade[1400]["points_ordered"]      # 7 点(与问题3同构, 漏检率高)
    p4_mid = trade[1100]["points_ordered"]        # 13 点
    p4_big = trade[1000]["points_ordered"]        # 19 点(漏检率 ~0.1%)

    N = int(os.environ.get("NTRIALS", "150"))
    out = {"config": {"n_trials": N}, "blocks": {}}

    plans = [
        (3, Params(name="A_adaptive"), p3_pts, "p3_A_adaptive"),
        (3, Params(name="B_sequential", interleave=False), p3_pts, "p3_B_sequential"),
        (3, Params(name="C_nosweep", use_sweep=False), p3_pts, "p3_C_nosweep"),
        (3, Params(name="D_notrack", use_tracking=False), p3_pts, "p3_D_notrack"),
        (4, Params(name="A_adaptive"), p4_big, "p4_A_cover19"),
        (4, Params(name="A_adaptive"), p4_mid, "p4_A_cover13"),
        (4, Params(name="A_adaptive"), p4_small, "p4_A_cover7"),
        (4, Params(name="B_sequential", interleave=False), p4_big, "p4_B_sequential"),
    ]
    for problem, params, pts, tag in plans:
        seed0 = 90000 if problem == 3 else 700000
        agg, _ = run_block(problem, params, pts, N, seed0, tag)
        out["blocks"][tag] = agg
        print(f"[{tag}] ratio={agg['mean_ratio']:.4f} full={agg['frac_full_clear']:.3f} "
              f"avgT={agg['mean_avg_time_s']:.1f}s total={agg['mean_total_time_s']:.0f}s "
              f"req={agg['mean_requests2']:.0f} anomalies={agg['mean_anomaly']:.2f} "
              f"({agg['wall_s']:.0f}s wall)", flush=True)
        with open(os.path.join(RES, "sim_summary.json"), "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    print("done")


if __name__ == "__main__":
    main()

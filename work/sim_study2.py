"""v2 改进的量化验证: 滚动时域重优化(RHO) / 朝向贝叶斯信念 / 风险可控自适应侦察。

对比组:
  v1        : 原策略(见 sim_study.py 的 A_adaptive 结果)
  v2-RHO    : 仅替换调度为滚动时域重优化(严格侦察点集不变)
  v2-RHO-B  : 再叠加问题4 朝向信念(问题3 无此项)
  v2-RISK   : 再叠加风险可控自适应侦察(theta 可调, 记录升级点数)
输出: results/sim2_summary.json, results/sim2_*.jsonl
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))
sys.path.insert(0, os.path.dirname(__file__))
from sim import generate_case  # noqa: E402
from strategy2 import Params2, run_trial_v2  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "results")
N = int(os.environ.get("NTRIALS", "120"))


def load_plans():
    cov = json.load(open(os.path.join(RES, "cover_points.json")))
    strict3 = cov["problem3"]["points_ordered"]
    trade = {round(t["spacing"]): t["points_ordered"] for t in cov["problem4_tradeoff"]}
    strict4 = trade[1000]
    cl = cov.get("cover_levels", {})
    lvl1 = cl.get("level1", {}).get("points_ordered", strict3[:5])
    esc = cl.get("escalation", {}).get("points_ordered", [])
    esc4 = [p for p in strict4
            if min(float(np.linalg.norm(np.array(p) - np.array(q))) for q in lvl1) > 80.0]
    # 风险可控自适应侦察: 仍用严格点集, 但按"风险收益/行程"排序并可提前停止(theta)
    return (strict3, strict4,
            {"level1": strict3, "escalation": []},
            {"level1": strict4, "escalation": []},
            {"level1": strict3, "escalation": []},
            {"level1": strict4, "escalation": []})


def run_block(problem, plan, params, tag, seed0):
    rows = []
    t0 = time.time()
    for i in range(N):
        rng = np.random.default_rng(seed0 + i)
        srcs = generate_case(rng, problem)
        r = run_trial_v2(srcs, plan, params, problem)
        rows.append(r)
    st = [r["stats"] for r in rows]
    ratio = np.array([s["ratio"] for s in st])
    avg = np.array([s["avg_time_s"] if s["avg_time_s"] else np.nan for s in st])
    out = {
        "tag": tag, "problem": problem, "n_trials": N,
        "mean_ratio": float(np.nanmean(ratio)),
        "frac_full_clear": float(np.mean(ratio >= 1 - 1e-9)),
        "mean_avg_time_s": float(np.nanmean(avg)),
        "median_avg_time_s": float(np.nanmedian(avg)),
        "p10_avg_time_s": float(np.nanpercentile(avg, 10)),
        "p90_avg_time_s": float(np.nanpercentile(avg, 90)),
        "mean_total_time_s": float(np.mean([s["total_time_s"] for s in st])),
        "mean_move_m": float(np.mean([s["move_m"] for s in st])),
        "mean_requests": float(np.mean([s["n_requests"] for s in st])),
        "mean_scan_points": float(np.mean([r["scan_points"] for r in rows])),
        "mean_esc_points": float(np.mean([r["n_esc_points"] for r in rows])),
        "frac_escalated": float(np.mean([r["escalated"] for r in rows])),
        "total_missed": int(np.sum([s["n_sources"] - s["n_cleared"] for s in st])),
        "wall_s": time.time() - t0,
        "time_breakdown": {k: float(np.mean([s[k] for s in st]))
                           for k in ("t_move", "t_switch", "t_measure", "t_clear")},
    }
    with open(os.path.join(RES, f"sim2_{tag}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({"stats": r["stats"], "cleared": r["cleared"],
                                "proven_empty": r["proven_empty"],
                                "unresolved": r["unresolved"],
                                "escalated": r["escalated"], "n_esc_points": r["n_esc_points"],
                                "scan_points": r["scan_points"],
                                "sources": r["sources"]}, ensure_ascii=False) + "\n")
    print(f"[{tag}] ratio={out['mean_ratio']:.4f} full={out['frac_full_clear']:.3f} "
          f"avgT={out['mean_avg_time_s']:.1f}s move={out['mean_move_m']:.0f}m "
          f"scans={out['mean_scan_points']:.1f} esc={out['mean_esc_points']:.1f} "
          f"req={out['mean_requests']:.0f} missed={out['total_missed']} ({out['wall_s']:.0f}s)",
          flush=True)
    return out


def main():
    s3, s4, rho3, rho4, risk3, risk4 = load_plans()
    out = {"config": {"n_trials": N}, "blocks": {}}
    prev = os.path.join(RES, "sim2_summary.json")
    if os.path.exists(prev) and os.environ.get("ONLY_TAGS"):
        out["blocks"] = json.load(open(prev))["blocks"]
    plans = [
        (3, rho3, Params2(name="v2_rho", theta_miss=0.0, use_belief=False), "p3_v2_rho", 90000),
        (3, rho3, Params2(name="v2_risk015", theta_miss=0.15, use_belief=False),
         "p3_v2_risk015", 90000),
        (3, rho3, Params2(name="v2_risk050", theta_miss=0.50, use_belief=False),
         "p3_v2_risk050", 90000),
        (4, rho4, Params2(name="v2_rho_nb", theta_miss=0.0, use_belief=False), "p4_v2_rho_nb", 700000),
        (4, rho4, Params2(name="v2_belief", theta_miss=0.0, use_belief=True), "p4_v2_belief", 700000),
        (4, rho4, Params2(name="v2_nb_risk", theta_miss=0.15, use_belief=False),
         "p4_v2_risk015_nb", 700000),
    ]
    only = os.environ.get("ONLY_TAGS")
    for problem, plan, params, tag, seed0 in plans:
        if only and tag not in only.split(","):
            continue
        out["blocks"][tag] = run_block(problem, plan, params, tag, seed0)
        with open(os.path.join(RES, "sim2_summary.json"), "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    print("done")


if __name__ == "__main__":
    main()

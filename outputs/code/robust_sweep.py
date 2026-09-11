"""系统性稳健性测试: 干扰源个数 N=10..20 × 全向/定向配比 {0,25,50,75,100}% 的全量多轮演练。

被测策略: v2 推荐配置(滚动时域重优化 + 朝向贝叶斯信念 + 严格环绕侦察, θ=0);
另设"全向专用"对照组(8 点覆盖侦察 + problem=3 路径)用于对比通用策略的代价。

输出:
  results/sweep_trials.jsonl    逐局完整明细(含每源清除时刻、首次发现时刻、失败类型)
  results/sweep_summary.json    单元格聚合(11×5=55 格)与逐源类型统计
  results/sweep_table.md        可直接引用的 Markdown 结果表
  results/sweep_p3_omni.jsonl   全向专用策略对照(仅 0% 列)
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))
from sim import R_ARENA, R_MAX, R_MIN, Simulator, Source  # noqa: E402
from strategy2 import Params2, RobotV2  # noqa: E402

RES = os.path.join(HERE, "results")
N_LIST = list(range(10, 21))
FRACS = [0.0, 0.25, 0.50, 0.75, 1.00]
TRIALS = int(os.environ.get("TRIALS", "30"))
SEED0 = 20260911
_PLANS: dict = {}


def load_plans():
    cov = json.load(open(os.path.join(RES, "cover_points.json")))
    s3 = cov["problem3"]["points_ordered"]
    p4 = {round(t["spacing"]): t["points_ordered"] for t in cov["problem4_tradeoff"]}[1000]
    return ({"level1": p4, "escalation": []}, {"level1": s3, "escalation": []})


def gen_case(seed, n, dir_frac):
    rng = np.random.default_rng(seed)
    chans = rng.choice(np.arange(1, 21), size=n, replace=False)
    n_dir = int(round(dir_frac * n))
    kinds = ["dir"] * n_dir + ["omni"] * (n - n_dir)
    rng.shuffle(kinds)
    srcs = []
    for i, ch in enumerate(chans):
        r = R_ARENA * np.sqrt(rng.random())
        a = 2 * np.pi * rng.random()
        pos = np.array([r * np.cos(a), r * np.sin(a)])
        rad = R_MIN + (R_MAX - R_MIN) * rng.random()
        kind = kinds[i]
        direction = None if kind == "omni" else float(rng.random() * 360.0)
        srcs.append(Source(int(ch), pos, rad, kind, direction))
    return srcs


def run_one(task):
    seed, n, frac, arm = task
    plan4, plan3 = _PLANS["p4"], _PLANS["p3"]
    srcs = gen_case(seed, n, frac)
    problem = 3 if arm == "p3" else 4
    plan = plan3 if arm == "p3" else plan4
    sim = Simulator(srcs)
    kind_of = {s.channel: s.kind for s in srcs}
    clear_events, detect_first = [], {}
    orig_clear, orig_measure = sim.clear, sim.measure

    def clear(pos, ch):
        res, t = orig_clear(pos, ch)
        if res == "success":
            clear_events.append([int(ch), round(sim.t, 3), kind_of[ch]])
        return res, t

    def measure(pos, ch):
        res, svd = orig_measure(pos, ch)
        if res in ("direction", "near") and ch not in detect_first:
            detect_first[int(ch)] = [round(sim.t, 3), kind_of[ch]]
        return res, svd

    sim.clear, sim.measure = clear, measure
    rb = RobotV2(sim, plan, Params2(theta_miss=0.0, use_belief=(problem == 4)), problem=problem)
    res = rb.run()
    st = res["stats"]
    tracked = set(res.get("tracked_channels", []))
    rec = {
        "arm": arm, "n": n, "dir_frac": frac, "seed": seed,
        "ratio": st["ratio"], "n_cleared": st["n_cleared"], "n_sources": st["n_sources"],
        "avg_time_s": st["avg_time_s"], "total_time_s": st["total_time_s"],
        "move_m": st["move_m"], "n_requests": st["n_requests"],
        "t_move": st["t_move"], "t_measure": st["t_measure"],
        "t_switch": st["t_switch"], "t_clear": st["t_clear"],
        "scan_points": len(rb.scanned), "esc_points": rb.n_esc_used,
        "anomaly": res.get("anomaly"), "n_gave_up": len(res.get("gave_up", [])),
        "cleared_events": clear_events,
        "detect_first": detect_first,
        "missed": [[int(s.channel), s.kind] for s in srcs if not s.cleared],
        "missed_never_detected": [int(s.channel) for s in srcs
                                  if not s.cleared and s.channel not in tracked],
        "omni_total": sum(1 for s in srcs if s.kind == "omni"),
        "omni_cleared": sum(1 for s in srcs if s.kind == "omni" and s.cleared),
        "dir_total": sum(1 for s in srcs if s.kind == "dir"),
        "dir_cleared": sum(1 for s in srcs if s.kind == "dir" and s.cleared),
    }
    return rec


def _init():
    _PLANS["p4"], _PLANS["p3"] = load_plans()


def main():
    workers = int(os.environ.get("WORKERS", "7"))
    tasks = []
    for n in N_LIST:
        for frac in FRACS:
            for i in range(TRIALS):
                seed = SEED0 + n * 100000 + int(round(frac * 100)) * 1000 + i
                tasks.append((seed, n, frac, "p4"))
    for n in N_LIST:                     # 全向专用策略对照
        for i in range(TRIALS):
            seed = SEED0 + n * 100000 + 900 + i
            tasks.append((seed, n, 0.0, "p3"))
    print(f"总任务 {len(tasks)} 局, 并行 {workers} 路", flush=True)

    t0 = time.time()
    rows, done = [], 0
    with Pool(workers, initializer=_init) as pool:
        for rec in pool.imap_unordered(run_one, tasks, chunksize=1):
            rows.append(rec)
            done += 1
            if done % 50 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"  {done}/{len(tasks)}  ({el:.0f}s, 预计剩余 "
                      f"{el / done * (len(tasks) - done):.0f}s)", flush=True)
    rows.sort(key=lambda r: (r["arm"], r["n"], r["dir_frac"], r["seed"]))
    with open(os.path.join(RES, "sweep_trials.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"逐局明细已写出: {len(rows)} 局, 用时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""生成论文用图 (中文标注)。输出到 work/figures/*.png"""
from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))
from geom import (  # noqa: E402
    diameter_rotating_calipers_pair, localization_polygon, mec, unit, wedge_halfplanes,
    clip_polygon, diameter_disk_covers,
)
from sim import generate_case, Simulator  # noqa: E402
from strategy import Params, Robot, run_trial  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "results")
FIG = os.path.join(os.path.dirname(__file__), "figures")
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
R_ARENA = 1800.0


def fig_p1_counterexample():
    with open(os.path.join(RES, "p1_summary.json")) as f:
        p1 = json.load(f)
    ce = p1["sharpest_counterexample"]
    S = [np.array(p) for p in ce["points"]]
    th = ce["thetas"]
    G = np.array(ce["G"])
    poly, _ = localization_polygon([(S[i], th[i], 1.0) for i in range(len(S))], arena=R_ARENA)
    D, pair = diameter_rotating_calipers_pair(poly)
    c, r = mec(poly)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    ax = axes[0]
    for i, s in enumerate(S):
        for hp in wedge_halfplanes(s, th[i], 1.0):
            pass
        u = unit(th[i])
        ax.plot([s[0], s[0] + 700 * u[0]], [s[1], s[1] + 700 * u[1]], "-", color="0.35", lw=1.0)
        for sgn in (+1, -1):
            d = unit(th[i] + sgn * 1.0)
            ax.plot([s[0], s[0] + 1400 * d[0]], [s[1], s[1] + 1400 * d[1]], ":",
                    color="0.6", lw=0.8)
        ax.plot(*s, "ko", ms=4)
        ax.annotate(rf"$S_{i+1}$", s + np.array([25, 25]), fontsize=9)
    P = np.array(poly + [poly[0]])
    ax.fill(P[:, 0], P[:, 1], color="#e8734a", alpha=0.55, zorder=3)
    ax.plot(P[:, 0], P[:, 1], color="#b03a12", lw=1.2, zorder=4)
    a, b = pair
    ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="#1f5fa8", lw=1.6, label="直径 $D$")
    thc = np.linspace(0, 2 * math.pi, 400)
    ax.plot((a[0] + b[0]) / 2 + (D / 2) * np.cos(thc), (a[1] + b[1]) / 2 + (D / 2) * np.sin(thc),
            "--", color="#1f5fa8", lw=1.4, label="以直径 $D$ 为直径的圆")
    ax.plot(c[0] + r * np.cos(thc), c[1] + r * np.sin(thc), "-", color="#12894d", lw=1.6,
            label=f"最小包围圆 $R^*={r:.1f}$m")
    ax.plot(*G, "r*", ms=11, label="真实干扰源 $G$")
    lo = np.array(poly).min(axis=0) - 120
    hi = np.array(poly).max(axis=0) + 120
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="best")
    ax.set_title(f"(a) 反例: $D$={D:.1f}m, $R^*$={r:.1f}m > $D/2$={D/2:.1f}m", fontsize=11)
    ax.set_xlabel("x / m"); ax.set_ylabel("y / m")

    ax2 = axes[1]
    for tag, lab, col in (("A", "检测点随机分布", "#1f5fa8"), ("B", "检测点环绕源", "#c0392b")):
        xs, ys = [], []
        for k in (2, 3, 4, 5):
            blk = p1[f"cover_{tag}_{k}"]
            xs.append(k); ys.append(blk["frac_covered_by_diameter_disk"] * 100)
        ax2.plot(xs, ys, "o-", color=col, label=lab)
        for x, y in zip(xs, ys):
            ax2.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(2, 6), fontsize=8)
    ax2.axhline(100, color="0.6", ls=":", lw=1)
    ax2.set_xticks([2, 3, 4, 5]); ax2.set_xlabel("检测点数 n")
    ax2.set_ylabel("直径圆覆盖定位区域的比例 / %")
    ax2.set_ylim(0, 105)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_title("(b) 直径圆能否覆盖: 分场景统计 (Jung 上界 $2/\\sqrt{3}$)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_p1_geometry.png"))
    plt.close(fig)


def fig_p1_dist():
    with open(os.path.join(RES, "p1_summary.json")) as f:
        p1 = json.load(f)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    labels = ["n=2", "n=3", "n=4", "n=5"]
    data = [[p1[f"cover_A_{k}"]["mean_ratio_2R_over_D"] for k in (2, 3, 4, 5)],
            [p1[f"cover_A_{k}"]["p95_ratio_2R_over_D"] for k in (2, 3, 4, 5)],
            [p1[f"cover_A_{k}"]["max_ratio_2R_over_D"] for k in (2, 3, 4, 5)]]
    x = np.arange(4)
    w = 0.26
    ax.bar(x - w, data[0], w, label="均值")
    ax.bar(x, data[1], w, label="95 分位")
    ax.bar(x + w, data[2], w, label="$2R^*/D$ 最大值")
    ax.axhline(2 / math.sqrt(3), color="r", ls="--", lw=1.2, label="Jung 理论上界 1.1547")
    ax.axhline(1.0, color="0.4", ls=":", lw=1.2, label="覆盖临界值 1.0")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("$2R^*/D$")
    ax.set_ylim(0.99, 1.18)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("定位区域最小包围圆半径与直径之比 (检测点随机分布)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_p1_ratio.png"))
    plt.close(fig)


def fig_p2():
    with open(os.path.join(RES, "p2_grid.json")) as f:
        grid = json.load(f)
    ds = sorted({g["d"] for g in grid})
    als = sorted({g["alpha"] for g in grid})
    Z = np.full((len(als), len(ds)), np.nan)
    P = np.full((len(als), len(ds)), np.nan)
    for g in grid:
        i = als.index(g["alpha"]); j = ds.index(g["d"])
        Z[i, j] = g["E_diam_geom"]
        P[i, j] = g["p_detect"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    ax = axes[0]
    im = ax.pcolormesh(ds, als, Z, shading="auto", cmap="viridis_r")
    cb = fig.colorbar(im, ax=ax); cb.set_label("$E[\\log D]$ 对应的几何均值直径 $E_g[D]$ / m")
    best = min(grid, key=lambda g: g["E_log_diam"])
    ax.plot(best["d"], best["alpha"], "r*", ms=14, label=f"最优: d={best['d']:.0f}m, α={best['alpha']:.0f}°")
    thr = best["E_log_diam"] + math.log(1.05)
    for g in grid:
        if g["E_log_diam"] <= thr:
            ax.plot(g["d"], g["alpha"], "ws", ms=5, mfc="none", mec="w")
    ax.set_xlabel("第二检测点距离 d / m"); ax.set_ylabel("相对首次示向度的偏角 α / °")
    ax.legend(fontsize=9); ax.set_title("(a) 期望定位区域尺度 $J(S_2)$ 的极坐标网格搜索")
    ax2 = axes[1]
    sc = ax2.scatter([g["p_detect"] for g in grid], [g["E_diam_detected"] for g in grid],
                     c=[g["alpha"] for g in grid], s=22, cmap="plasma")
    cb2 = fig.colorbar(sc, ax=ax2); cb2.set_label("偏角 α / °")
    front = []
    for g in sorted([g for g in grid if g["E_diam_detected"]], key=lambda g: -g["E_diam_detected"]):
        if not front or g["p_detect"] > front[-1]["p_detect"] + 1e-9:
            front.append(g)
    ax2.plot([g["p_detect"] for g in front], [g["E_diam_detected"] for g in front], "k--", lw=1.4,
             label="Pareto 前沿")
    ax2.set_xlabel("第二点可测到信号的概率 $P_{det}$")
    ax2.set_ylabel("命中情形下定位区域直径均值 $E[D|det]$ / m")
    ax2.set_yscale("log"); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    ax2.set_title("(b) 探测概率-定位精度权衡")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_p2_second_point.png"))
    plt.close(fig)


def _draw_cover(ax, pts, color, label, route=True):
    P = np.array(pts)
    for p in P:
        ax.add_patch(plt.Circle(p, 1000.0, color=color, alpha=0.06, lw=0))
    if route:
        ax.plot(P[:, 0], P[:, 1], "-", color=color, lw=1.2, alpha=0.8, label=label)
    ax.plot(P[:, 0], P[:, 1], "o", color=color, ms=5)
    ax.plot(0, 0, "k^", ms=8, label="起点(0,0)")


def fig_cover():
    with open(os.path.join(RES, "cover_points.json")) as f:
        cov = json.load(f)
    p3 = cov["problem3"]["points_ordered"]
    trade = {round(t["spacing"]): t for t in cov["problem4_tradeoff"]}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    for ax, pts, ttl in ((axes[0], p3, "(a) 问题3: 8 个侦察点(覆盖半径 ≤950m, 巡线 6.7km)"),
                         (axes[1], trade[1000]["points_ordered"],
                          "(b) 问题4: 19 个侦察点(环绕条件, 漏检率 0.13%)")):
        ax.add_patch(plt.Circle((0, 0), R_ARENA, color="k", fill=False, lw=1.0))
        ax.add_patch(plt.Circle((0, 0), R_ARENA, color="0.85", alpha=0.35, lw=0))
        _draw_cover(ax, pts, "#1f5fa8" if ax is axes[0] else "#12894d", "侦察巡线")
        ax.set_aspect("equal"); ax.set_xlim(-2600, 2600); ax.set_ylim(-2600, 2600)
        ax.set_title(ttl, fontsize=11); ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_cover_points.png"))
    plt.close(fig)


def fig_trajectory(problem=3, seed=90003):
    cov = json.load(open(os.path.join(RES, "cover_points.json")))
    pts = cov["problem3"]["points_ordered"] if problem == 3 else \
        {round(t["spacing"]): t["points_ordered"] for t in cov["problem4_tradeoff"]}[1000]
    rng = np.random.default_rng(seed)
    srcs = generate_case(rng, problem)
    sim = Simulator(srcs)
    rb = Robot(sim, pts, Params(name="A"), problem=problem)
    path = [sim.pos.copy()]
    orig_measure, orig_clear = sim.measure, sim.clear

    def m(pos, ch):
        r = orig_measure(pos, ch)
        path.append(sim.pos.copy())
        return r

    def c(pos, ch):
        r = orig_clear(pos, ch)
        path.append(sim.pos.copy())
        return r

    sim.measure, sim.clear = m, c
    rb.run()
    P = np.array(path)
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.add_patch(plt.Circle((0, 0), R_ARENA, color="0.9", alpha=0.5, lw=0))
    ax.add_patch(plt.Circle((0, 0), R_ARENA, color="k", fill=False, lw=1.0))
    ax.plot(P[:, 0], P[:, 1], "-", color="#1f5fa8", lw=0.9, alpha=0.85, label="机器狗轨迹")
    for s in srcs:
        ax.plot(*s.pos, "o", color="#c0392b" if s.kind == "dir" else "#e67e22", ms=6,
                label=("定向源" if s.kind == "dir" else "全向源"))
        ax.annotate(f"ch{s.channel}", s.pos + np.array([25, 25]), fontsize=7)
    ax.plot(0, 0, "k^", ms=9, label="起点")
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="upper right")
    ax.set_aspect("equal"); ax.set_xlim(-2000, 2000); ax.set_ylim(-2000, 2000)
    ax.set_xlabel("x / m"); ax.set_ylabel("y / m")
    st = sim.stats()
    ax.set_title(f"问题{problem} 一次演练轨迹 (N={st['n_sources']}, 清除 {st['n_cleared']}, "
                 f"总时间 {st['total_time_s']:.0f}s, 平均 {st['avg_time_s']:.0f}s)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig_traj_p{problem}.png"))
    plt.close(fig)


def fig_strategy_compare():
    with open(os.path.join(RES, "sim_summary.json")) as f:
        d = json.load(f)
    blocks = d["blocks"]
    order = [("p3_A_adaptive", "问题3 A\n插序+追踪+扫掠"),
             ("p3_B_sequential", "问题3 B\n先侦察后清除"),
             ("p3_C_nosweep", "问题3 C\n不用清除扫描"),
             ("p3_D_notrack", "问题3 D\n扫掠优先"),
             ("p4_A_cover19", "问题4 A\n19 点侦察集"),
             ("p4_B_sequential", "问题4 B\n19 点侦察集"),
             ("p4_A_cover13", "问题4 A\n13 点侦察集"),
             ("p4_A_cover7", "问题4 A\n7 点侦察集")]
    labels = [lab for key, lab in order if key in blocks]
    keys = [key for key, lab in order if key in blocks]
    times = [blocks[k]["mean_avg_time_s"] for k in keys]
    ratios = [blocks[k]["mean_ratio"] * 100 for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    col = ["#1f5fa8"] * 4 + ["#12894d"] * 4
    axes[0].bar(range(len(keys)), times, color=col[:len(keys)])
    axes[0].set_xticks(range(len(keys)))
    axes[0].set_xticklabels(labels, fontsize=7.5, rotation=18, ha="right")
    axes[0].set_ylabel("平均定位清除时间 / s")
    for i, v in enumerate(times):
        axes[0].annotate(f"{v:.0f}", (i, v), textcoords="offset points", xytext=(0, 3),
                         ha="center", fontsize=8)
    axes[0].set_title("(a) 平均定位清除时间(越小越好)", fontsize=11)
    axes[1].bar(range(len(keys)), ratios, color=col[:len(keys)])
    axes[1].set_xticks(range(len(keys)))
    axes[1].set_xticklabels(labels, fontsize=7.5, rotation=18, ha="right")
    axes[1].set_ylim(95, 100.6); axes[1].set_ylabel("被清除干扰源个数比例 / %")
    for i, v in enumerate(ratios):
        axes[1].annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 3),
                         ha="center", fontsize=8)
    axes[1].set_title("(b) 清除比例(越大越好)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_strategy_compare.png"))
    plt.close(fig)


def fig_p4_tradeoff():
    cov = json.load(open(os.path.join(RES, "cover_points.json")))
    tr = cov["problem4_tradeoff"]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    seen = {}
    for t in tr:
        key = (t["spacing"], t["margin"])
        seen[key] = t
    xs = [t["n_points"] for t in seen.values()]
    ys = [t["mean_miss_prob"] * 100 for t in seen.values()]
    cs = [t["route_length_m"] / 1000.0 for t in seen.values()]
    sc = ax.scatter(xs, ys, c=cs, s=70, cmap="viridis")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("巡线长度 / km")
    for t in seen.values():
        ax.annotate(f"s={t['spacing']:.0f}", (t["n_points"], t["mean_miss_prob"] * 100),
                    textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_xlabel("侦察点数"); ax.set_ylabel("平均漏检方向比例 $\\bar f$ / %")
    ax.set_yscale("symlog", linthresh=1)
    ax.grid(alpha=0.3)
    ax.set_title("问题4 定向源侦察代价-漏检率权衡(环绕条件)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_p4_tradeoff.png"))
    plt.close(fig)


def fig_time_breakdown():
    with open(os.path.join(RES, "sim_summary.json")) as f:
        d = json.load(f)
    blocks = d["blocks"]
    keys = ["p3_A_adaptive", "p3_B_sequential", "p3_C_nosweep", "p3_D_notrack",
            "p4_A_cover19", "p4_B_sequential", "p4_A_cover13", "p4_A_cover7"]
    keys = [k for k in keys if k in blocks]
    labels = [k.replace("_", "\n") for k in keys]
    comp = ["move", "measure", "switch", "clear"]
    names = ["移动", "检测", "换频道", "清除"]
    cols = ["#1f5fa8", "#e67e22", "#95a5a6", "#c0392b"]
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    bottom = np.zeros(len(keys))
    for c, nm, cl in zip(comp, names, cols):
        vals = np.array([blocks[k]["time_breakdown"][c] for k in keys])
        ax.bar(range(len(keys)), vals, bottom=bottom, label=nm, color=cl)
        bottom += vals
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("平均总时间 / s (每局)")
    ax.legend(ncol=4, fontsize=9)
    ax.set_title("总时间构成(移动时间占主导)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_time_breakdown.png"))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_p1_counterexample()
    fig_p1_dist()
    fig_p2()
    fig_cover()
    fig_trajectory(3, 90003)
    fig_trajectory(4, 700003)
    fig_strategy_compare()
    fig_p4_tradeoff()
    fig_time_breakdown()
    print("figures done:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()

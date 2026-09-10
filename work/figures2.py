"""v2 文档新增插图。"""
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
from cover_design import arena_grid, max_cover_radius  # noqa: E402
from strategy2 import DirBelief, MissModel, Params2, RobotV2  # noqa: E402
from sim import R_ARENA, Simulator, generate_case, Source  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "results")
FIG = os.path.join(os.path.dirname(__file__), "figures")
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


def _blocks(path):
    with open(path) as f:
        return json.load(f)["blocks"]


def fig_compare():
    v1 = _blocks(os.path.join(RES, "sim_summary.json"))
    v2 = _blocks(os.path.join(RES, "sim2_summary.json"))
    rows = [
        ("问题3", v1["p3_A_adaptive"], v2.get("p3_v2_rho")),
        ("问题4", v1.get("p4_A_cover19"), v2.get("p4_v2_belief")),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6))
    labels = [r[0] for r in rows]
    x = np.arange(len(rows))
    w = 0.34
    t1 = [r[1]["mean_avg_time_s"] for r in rows]
    t2 = [r[2]["mean_avg_time_s"] if r[2] else np.nan for r in rows]
    axes[0].bar(x - w / 2, t1, w, label="v1", color="#9aa7b4")
    axes[0].bar(x + w / 2, t2, w, label="v2", color="#1f5fa8")
    for xi, (a, b) in enumerate(zip(t1, t2)):
        axes[0].annotate(f"{a:.0f}", (xi - w / 2, a), textcoords="offset points",
                         xytext=(0, 3), ha="center", fontsize=9)
        axes[0].annotate(f"{b:.0f}", (xi + w / 2, b), textcoords="offset points",
                         xytext=(0, 3), ha="center", fontsize=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("平均定位清除时间 / s"); axes[0].legend()
    axes[0].set_title("(a) 平均定位清除时间: v2 全面优于 v1", fontsize=11)

    r1 = [r[1]["mean_ratio"] * 100 for r in rows]
    r2 = [r[2]["mean_ratio"] * 100 if r[2] else np.nan for r in rows]
    axes[1].bar(x - w / 2, r1, w, label="v1", color="#9aa7b4")
    axes[1].bar(x + w / 2, r2, w, label="v2", color="#12894d")
    for xi, (a, b) in enumerate(zip(r1, r2)):
        axes[1].annotate(f"{a:.2f}", (xi - w / 2, a), textcoords="offset points",
                         xytext=(0, 3), ha="center", fontsize=9)
        axes[1].annotate(f"{b:.2f}", (xi + w / 2, b), textcoords="offset points",
                         xytext=(0, 3), ha="center", fontsize=9)
    axes[1].set_ylim(95, 100.5); axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("清除比例 / %"); axes[1].legend()
    axes[1].set_title("(b) 清除比例: 保持 100%", fontsize=11)

    mv1 = [r[1]["mean_move_m"] / 1000 for r in rows]
    mv2 = [r[2]["mean_move_m"] / 1000 if r[2] else np.nan for r in rows]
    axes[2].bar(x - w / 2, mv1, w, label="v1", color="#9aa7b4")
    axes[2].bar(x + w / 2, mv2, w, label="v2", color="#e67e22")
    for xi, (a, b) in enumerate(zip(mv1, mv2)):
        axes[2].annotate(f"{a:.1f}", (xi - w / 2, a), textcoords="offset points",
                         xytext=(0, 3), ha="center", fontsize=9)
        axes[2].annotate(f"{b:.1f}", (xi + w / 2, b), textcoords="offset points",
                         xytext=(0, 3), ha="center", fontsize=9)
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("平均移动距离 / km"); axes[2].legend()
    axes[2].set_title("(c) 行程: 滚动重优化显著缩短", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_v2_compare.png"))
    plt.close(fig)


def fig_risk():
    cov = json.load(open(os.path.join(RES, "cover_points.json")))
    pts = np.array(cov["problem3"]["points_ordered"])
    mm = MissModel(problem=3)
    ks, ems, mus = [], [], []
    for k in range(1, len(pts) + 1):
        ks.append(k)
        ems.append(mm.expected_missed(pts[:k], n_unresolved=7, n_detected=13))
        mus.append(mm.mu(pts[:k]))
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    ax = axes[0]
    ax.plot(ks, ems, "o-", color="#1f5fa8", label="期望漏检源个数 $E[\\mu N]$")
    ax.set_xlabel("已执行侦察点数 k"); ax.set_ylabel("期望漏检源个数", color="#1f5fa8")
    ax.set_yscale("symlog", linthresh=0.1)
    ax2 = ax.twinx()
    ax2.plot(ks, [m * 100 for m in mus], "s--", color="#c0392b", label="漏检概率 $\\mu$ / %")
    ax2.set_ylabel("随机源漏检概率 $\\mu$ / %", color="#c0392b")
    for th, c in ((0.15, "#e67e22"), (0.50, "#12894d")):
        ax.axhline(th, color=c, ls=":", lw=1.1)
        ax.annotate(f"$\\theta$={th}", (1.1, th), color=c, fontsize=9)
    ax.set_title("(a) 残余风险随侦察点数递减(8 点即严格覆盖)", fontsize=11)
    ax.grid(alpha=0.3)

    v2 = _blocks(os.path.join(RES, "sim2_summary.json"))
    keys = [("p3_v2_rho", "$\\theta$=0(严格)"), ("p3_v2_risk015", "$\\theta$=0.15"),
            ("p3_v2_risk050", "$\\theta$=0.5")]
    xs = [v2[k]["mean_ratio"] * 100 for k, _ in keys if k in v2]
    ys = [v2[k]["mean_avg_time_s"] for k, _ in keys if k in v2]
    ls = [lab for k, lab in keys if k in v2]
    ax3 = axes[1]
    ax3.plot(xs, ys, "o-", color="#1f5fa8")
    for a, b, l in zip(xs, ys, ls):
        ax3.annotate(l, (a, b), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax3.set_xlabel("清除比例 / %"); ax3.set_ylabel("平均定位清除时间 / s")
    ax3.grid(alpha=0.3)
    ax3.set_title("(b) 问题3: 风险阈值 $\\theta$ 的实测权衡", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_v2_risk.png"))
    plt.close(fig)


def fig_belief():
    """用一次真实演练展示朝向信念: 定向源背向盲区与前向侧取数。"""
    rng = np.random.default_rng(700003)
    srcs = generate_case(rng, 4)
    tgt = next(s for s in srcs if s.kind == "dir")
    sim = Simulator(srcs)
    cov = json.load(open(os.path.join(RES, "cover_points.json")))
    p4 = {round(t["spacing"]): t["points_ordered"] for t in cov["problem4_tradeoff"]}[1000]
    rb = RobotV2(sim, {"level1": p4, "escalation": []}, Params2(), problem=4)
    path = [sim.pos.copy()]
    om, oc = sim.measure, sim.clear

    def m(pos, ch):
        r = om(pos, ch)
        path.append(sim.pos.copy())
        return r

    def c(pos, ch):
        r = oc(pos, ch)
        path.append(sim.pos.copy())
        return r

    sim.measure, sim.clear = m, c
    rb.run()
    P = np.array(path)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
    ax = axes[0]
    ax.add_patch(plt.Circle((0, 0), R_ARENA, color="0.92", alpha=0.6, lw=0))
    ax.add_patch(plt.Circle((0, 0), R_ARENA, color="k", fill=False, lw=1.0))
    ax.plot(P[:, 0], P[:, 1], "-", color="#1f5fa8", lw=1.0, label="机器狗轨迹")
    for s in srcs:
        ax.plot(*s.pos, "o", color="#c0392b" if s.kind == "dir" else "#e67e22", ms=6)
        if s.kind == "dir":
            u = np.array([math.cos(math.radians(s.direction_deg)),
                          math.sin(math.radians(s.direction_deg))])
            n = np.array([-u[1], u[0]])
            ang = np.linspace(0, math.pi, 60)
            pts = np.array([s.pos + 600 * (math.cos(a) * u + math.sin(a) * n) for a in ang])
            ax.fill(np.concatenate([pts[:, 0], [s.pos[0]]]),
                    np.concatenate([pts[:, 1], [s.pos[1]]]), color="#c0392b", alpha=0.10, lw=0)
            ax.annotate("", xy=s.pos + 500 * u, xytext=s.pos,
                        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.6))
    ax.plot(0, 0, "k^", ms=9)
    ax.set_aspect("equal"); ax.set_xlim(-2000, 2000); ax.set_ylim(-2000, 2000)
    ax.set_xlabel("x / m"); ax.set_ylabel("y / m")
    ax.set_title("(a) 定向源覆盖半平面(淡红)与前向/背向", fontsize=11)

    ax = axes[1]
    nu = 120
    angs = 2 * np.pi * np.arange(nu) / nu
    b = DirBelief(nu=nu)
    box = [np.array([-900, -900]), np.array([1900, -900]),
           np.array([1900, 1900]), np.array([-900, 1900])]
    b.add(np.array([0.0, 0.0]), "direction", 0.0)
    r1 = b.evaluate(box)
    b.add(np.array([700.0, 200.0]), "no_signal")
    r2 = b.evaluate(box)
    for res, col, lab, ls in ((r1, "#1f5fa8", "仅 1 次 direction", "-"),
                              (r2, "#12894d", "+ 1 次 no_signal", "--")):
        if res is None or "u_post" not in res:
            continue
        ax.plot(angs, res["u_post"], ls, color=col, lw=1.8, label=lab)
    ax.set_title("(b) 朝向 $u$ 的后验: no_signal 把 180° 不确定压缩为窄弧", fontsize=11, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_v2_belief.png"))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_compare()
    fig_risk()
    fig_belief()
    print("v2 figures:", sorted(f for f in os.listdir(FIG) if f.startswith("fig_v2")))


if __name__ == "__main__":
    main()

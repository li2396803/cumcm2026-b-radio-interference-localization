"""汇总 robust_sweep.py 的逐局明细: 单元格统计表、按源类型统计、失败分类与插图。"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

matplotlib_ok = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti TC", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
except Exception:
    matplotlib_ok = False

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
N_LIST = list(range(10, 21))
FRACS = [0.0, 0.25, 0.50, 0.75, 1.00]


def load():
    rows = [json.loads(l) for l in open(os.path.join(RES, "sweep_trials.jsonl"))]
    return rows


def agg(cell):
    """cell: list of trial records → 聚合统计。"""
    n = len(cell)
    ratio = np.array([r["ratio"] for r in cell])
    avg = np.array([r["avg_time_s"] if r["avg_time_s"] else np.nan for r in cell])
    tot = np.array([r["total_time_s"] for r in cell])
    mv = np.array([r["move_m"] for r in cell])
    req = np.array([r["n_requests"] for r in cell])
    sc = [r["scan_points"] for r in cell]
    sc = np.array([x for x in sc if x is not None]) if any(x is not None for x in sc) \
        else np.array([19.0 if cell[0]["arm"] == "p4" else 8.0])   # 严格模式: 访问全部计划点
    omni_t = sum(r["omni_total"] for r in cell)
    omni_c = sum(r["omni_cleared"] for r in cell)
    dir_t = sum(r["dir_total"] for r in cell)
    dir_c = sum(r["dir_cleared"] for r in cell)
    never = sum(len(r["missed_never_detected"]) for r in cell)
    detected_fail = sum(len(r["missed"]) - len(r["missed_never_detected"]) for r in cell)
    return {
        "n_trials": n,
        "ratio": float(np.nanmean(ratio)),
        "min_ratio": float(np.nanmin(ratio)),
        "full_clear_rate": float(np.mean(ratio >= 1 - 1e-9)),
        "n_full_clear": int(np.sum(ratio >= 1 - 1e-9)),
        "avg_time_mean": float(np.nanmean(avg)),
        "avg_time_median": float(np.nanmedian(avg)),
        "avg_time_p10": float(np.nanpercentile(avg, 10)),
        "avg_time_p90": float(np.nanpercentile(avg, 90)),
        "avg_time_std": float(np.nanstd(avg)),
        "total_time_mean": float(np.mean(tot)),
        "move_mean": float(np.mean(mv)),
        "req_mean": float(np.mean(req)),
        "scan_mean": float(np.mean(sc)),
        "omni_total": omni_t, "omni_cleared": omni_c,
        "dir_total": dir_t, "dir_cleared": dir_c,
        "omni_rate": omni_c / omni_t if omni_t else None,
        "dir_rate": dir_c / dir_t if dir_t else None,
        "missed_never_detected": never,
        "missed_detected_failed": detected_fail,
        "t_move": float(np.mean([r["t_move"] for r in cell])),
        "t_measure": float(np.mean([r["t_measure"] for r in cell])),
        "t_switch": float(np.mean([r["t_switch"] for r in cell])),
        "t_clear": float(np.mean([r["t_clear"] for r in cell])),
    }


def safe_mean(a):
    a = np.asarray(a, dtype=float)
    return float(a.mean()) if a.size else float("nan")


def pct(x, nd=2):
    return "-" if x is None else f"{x*100:.{nd}f}%"


def interval_stats(cell, kind):
    """同一局内相邻两次成功清除的时间间隔(按源类型)。"""
    out = []
    for r in cell:
        ev = sorted(r["cleared_events"], key=lambda e: e[1])
        prev = 0.0
        for ch, t, k in ev:
            if k == kind:
                out.append(t - prev)
            prev = t
    return np.array(out) if out else np.array([], dtype=float)


def first_detect_stats(cell, kind):
    vals = []
    for r in cell:
        for ch, (t, k) in r["detect_first"].items():
            if k == kind:
                vals.append(t)
    return np.array(vals, dtype=float)


def main():
    rows = load()
    p4 = [r for r in rows if r["arm"] == "p4"]
    p3 = [r for r in rows if r["arm"] == "p3"]
    summary = {"config": {"N": N_LIST, "dir_fracs": FRACS,
                          "trials_per_cell": len({r["seed"] for r in p4 if r["n"] == 10
                                                  and r["dir_frac"] == 0.0}),
                          "strategy": "v2 = RHO + 朝向信念 + 严格环绕侦察(θ=0)"},
               "cells": {}, "by_kind": {}, "p3_omni": {}}
    for n in N_LIST:
        for f in FRACS:
            cell = [r for r in p4 if r["n"] == n and abs(r["dir_frac"] - f) < 1e-9]
            if not cell:
                continue
            key = f"{n}|{f:.2f}"
            summary["cells"][key] = agg(cell)
            if f > 0:
                summary["by_kind"][key] = {
                    "omni_interval_mean": float(np.nanmean(interval_stats(cell, "omni"))),
                    "dir_interval_mean": float(np.nanmean(interval_stats(cell, "dir"))),
                    "omni_first_detect_mean": float(np.nanmean(first_detect_stats(cell, "omni"))),
                    "dir_first_detect_mean": float(np.nanmean(first_detect_stats(cell, "dir"))),
                }
        cell3 = [r for r in p3 if r["n"] == n]
        if cell3:
            summary["p3_omni"][str(n)] = agg(cell3)
    with open(os.path.join(RES, "sweep_summary.json"), "w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # ---------- Markdown 表 ----------
    def grid(fmt, field, scale=1.0):
        head = "| N \\ 定向占比 | " + " | ".join(f"{f:.0%}" for f in FRACS) + " |"
        sep = "|---|" + "---|" * len(FRACS)
        lines = [head, sep]
        for n in N_LIST:
            cells = []
            for f in FRACS:
                c = summary["cells"].get(f"{n}|{f:.2f}")
                cells.append("-" if c is None else fmt.format(c[field] * scale))
            lines.append(f"| **{n}** | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    t_ratio = grid("{:.2f}", "ratio", 100)
    t_time = grid("{:.0f}", "avg_time_mean")
    t_full = grid("{:.0f}", "full_clear_rate", 100)
    t_move = grid("{:.1f}", "move_mean", 0.001)
    t_req = grid("{:.0f}", "req_mean")

    kind_lines = ["| N | 定向占比 | 全向清除率 | 定向清除率 | 全向平均清除间隔 | 定向平均清除间隔 | 全向首次发现 | 定向首次发现 |",
                  "|---|---|---|---|---|---|---|---|"]
    for n in N_LIST:
        for f in FRACS:
            if f == 0:
                continue
            key = f"{n}|{f:.2f}"
            c = summary["cells"].get(key)
            k = summary["by_kind"].get(key)
            if not c or not k:
                continue
            def num(x):
                return "-" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.0f} s"
            kind_lines.append(
                f"| {n} | {f:.0%} | {pct(c['omni_rate'])} | {pct(c['dir_rate'])} | "
                f"{num(k['omni_interval_mean'])} | {num(k['dir_interval_mean'])} | "
                f"{num(k['omni_first_detect_mean'])} | {num(k['dir_first_detect_mean'])} |")
    t_kind = "\n".join(kind_lines)

    comp = ["| N | 通用策略(19点环绕) 平均时间 | 通用策略比例 | 全向专用(8点覆盖) 平均时间 | 全向专用比例 | 通用策略代价 |",
            "|---|---|---|---|---|---|"]
    for n in N_LIST:
        c4 = summary["cells"].get(f"{n}|0.00")
        c3 = summary["p3_omni"].get(str(n))
        if not c4 or not c3:
            continue
        comp.append(f"| {n} | {c4['avg_time_mean']:.0f} s | {c4['ratio']*100:.2f}% | "
                    f"{c3['avg_time_mean']:.0f} s | {c3['ratio']*100:.2f}% | "
                    f"+{c4['avg_time_mean']-c3['avg_time_mean']:.0f} s "
                    f"(+{(c4['avg_time_mean']/c3['avg_time_mean']-1)*100:.0f}%) |")
    t_comp = "\n".join(comp)

    md = f"""# 系统性多轮测试结果（N=10…20 × 全向/定向配比）

被测策略：**v2 推荐配置**（滚动时域重优化 + 朝向贝叶斯信念 + 严格环绕侦察，θ=0）；
每格 {summary['config']['trials_per_cell']} 局，随机种子固定；有效接收半径 R~U(1000,1500) m、
位置在目标圆域内均匀、定向源朝向随机。

## 表 1 被清除干扰源个数比例 / %

{t_ratio}

## 表 2 全清局数比例 / %

{t_full}

## 表 3 平均定位清除时间 / s（每源）

{t_time}

## 表 4 平均移动距离 / km（每局）

{t_move}

## 表 5 平均请求数（每局）

{t_req}

## 表 6 按源类型分解（清除率 / 平均清除间隔 / 平均首次发现时刻）

{t_kind}

## 表 7 全向专用策略 vs 通用策略（0% 定向列）

{t_comp}
"""
    with open(os.path.join(RES, "sweep_table.md"), "w") as fh:
        fh.write(md)
    print(md[:1500])

    # ---------- 插图 ----------
    if not matplotlib_ok:
        print("matplotlib 不可用, 跳过出图")
        return
    os.makedirs(FIG, exist_ok=True)
    grid_ratio = np.full((len(N_LIST), len(FRACS)), np.nan)
    grid_time = np.full_like(grid_ratio, np.nan)
    for i, n in enumerate(N_LIST):
        for j, f in enumerate(FRACS):
            c = summary["cells"].get(f"{n}|{f:.2f}")
            if c:
                grid_ratio[i, j] = c["ratio"] * 100
                grid_time[i, j] = c["avg_time_mean"]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    for ax, g, ttl, cmap, fmt in (
            (axes[0], grid_ratio, "(a) 被清除干扰源个数比例 / %", "RdYlGn", "{:.1f}"),
            (axes[1], grid_time, "(b) 平均定位清除时间 / s（每源）", "viridis_r", "{:.0f}")):
        im = ax.imshow(g, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(FRACS)))
        ax.set_xticklabels([f"{f:.0%}" for f in FRACS])
        ax.set_yticks(range(len(N_LIST)))
        ax.set_yticklabels(N_LIST)
        ax.set_xlabel("定向干扰源占比"); ax.set_ylabel("干扰源个数 N")
        ax.set_title(ttl, fontsize=11)
        for i in range(len(N_LIST)):
            for j in range(len(FRACS)):
                if not math.isnan(g[i, j]):
                    ax.text(j, i, fmt.format(g[i, j]), ha="center", va="center", fontsize=7.5)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sweep_grid.png"))
    plt.close(fig)

    # 折线: 按配比的 N-曲线
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6))
    for j, f in enumerate(FRACS):
        ys = [summary["cells"].get(f"{n}|{f:.2f}", {}).get("avg_time_mean", np.nan) for n in N_LIST]
        axes[0].plot(N_LIST, ys, "o-", label=f"{f:.0%} 定向")
        ys2 = [summary["cells"].get(f"{n}|{f:.2f}", {}).get("ratio", np.nan) * 100 for n in N_LIST]
        axes[1].plot(N_LIST, ys2, "o-", label=f"{f:.0%} 定向")
    axes[0].set_xlabel("干扰源个数 N"); axes[0].set_ylabel("平均定位清除时间 / s")
    axes[1].set_xlabel("干扰源个数 N"); axes[1].set_ylabel("被清除源比例 / %")
    axes[1].set_ylim(90, 100.6)
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_title("(a) 平均定位清除时间随 N 与定向占比的变化", fontsize=11)
    axes[1].set_title("(b) 清除比例随 N 与定向占比的变化", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sweep_curves.png"))
    plt.close(fig)

    # 按类型: 清除率与清除间隔
    def val(n, f, key, scale=1.0):
        c = summary["cells"].get(f"{n}|{f:.2f}")
        if not c:
            return np.nan
        v = c.get(key)
        return np.nan if v is None else v * scale

    def kval(n, f, key):
        k = summary["by_kind"].get(f"{n}|{f:.2f}")
        if not k:
            return np.nan
        v = k.get(key)
        return np.nan if v is None else v

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.4))
    for j, f in enumerate(FRACS):
        if f == 0:
            continue
        axes[0].plot(N_LIST, [val(n, f, "omni_rate", 100) for n in N_LIST], "o-",
                     color="#e67e22", alpha=0.30, lw=1.0)
        axes[0].plot(N_LIST, [val(n, f, "dir_rate", 100) for n in N_LIST], "s--",
                     color="#c0392b", alpha=0.30, lw=1.0)
    axes[0].plot(N_LIST, [val(n, 0.50, "omni_rate", 100) for n in N_LIST], "o-",
                 color="#e67e22", lw=2.2, label="全向源（50% 配比）")
    axes[0].plot(N_LIST, [val(n, 0.50, "dir_rate", 100) for n in N_LIST], "s--",
                 color="#c0392b", lw=2.2, label="定向源（50% 配比）")
    axes[0].set_xlabel("干扰源个数 N"); axes[0].set_ylabel("清除率 / %")
    axes[0].set_ylim(90, 100.6); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[0].set_title("(a) 两类源的清除率（细线为其它配比）", fontsize=11)
    axes[1].plot(N_LIST, [kval(n, 0.50, "omni_interval_mean") for n in N_LIST], "o-",
                 color="#e67e22", label="全向源 清除间隔")
    axes[1].plot(N_LIST, [kval(n, 0.50, "dir_interval_mean") for n in N_LIST], "s--",
                 color="#c0392b", label="定向源 清除间隔")
    axes[1].plot(N_LIST, [kval(n, 0.50, "omni_first_detect_mean") for n in N_LIST], "o:",
                 color="#e67e22", label="全向源 首次发现时刻")
    axes[1].plot(N_LIST, [kval(n, 0.50, "dir_first_detect_mean") for n in N_LIST], "s:",
                 color="#c0392b", label="定向源 首次发现时刻")
    axes[1].set_xlabel("干扰源个数 N"); axes[1].set_ylabel("时间 / s")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    axes[1].set_title("(b) 两类源的清除间隔与首次发现时刻（50% 配比）", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sweep_kind.png"))
    plt.close(fig)
    print("图已输出: fig_sweep_grid.png / fig_sweep_curves.png / fig_sweep_kind.png")


if __name__ == "__main__":
    main()

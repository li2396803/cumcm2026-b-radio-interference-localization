"""检测覆盖点设计

问题3(全向源): 侦察点集的 1000m 圆盘并集须覆盖整个目标圆域 -> 覆盖半径 <= 950m(5% 余量)。
问题4(定向源): 更强的"环绕条件": 对目标圆域内任意 q, 所有距 q 不超过 1000m 的侦察点
   必须"包围" q(即 q 落在这些点的凸包内, 等价于相邻方向最大间隔 < 180 度)。
   这样无论定向源朝向如何, 至少有一个侦察点位于其覆盖半平面内。
输出: results/cover_points.json
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

R_ARENA = 1800.0
R_SAFE = 950.0
R_SURR = 1000.0
RES = os.path.join(os.path.dirname(__file__), "results")


def arena_grid(step=4.0, margin=0.0):
    xs = np.arange(-R_ARENA - margin, R_ARENA + margin + step, step)
    X, Y = np.meshgrid(xs, xs)
    m = (X ** 2 + Y ** 2) <= (R_ARENA + margin) ** 2
    return np.stack([X[m], Y[m]], axis=1)


def hex_lattice(spacing, margin=0.0):
    dy = spacing * math.sqrt(3) / 2.0
    pts = []
    ny = int((R_ARENA + margin) / dy) + 2
    for r in range(-ny, ny + 1):
        y = r * dy
        x0 = (spacing / 2.0) if (r % 2) else 0.0
        nx = int((R_ARENA + margin) / spacing) + 2
        for c in range(-nx, nx + 1):
            x = x0 + c * spacing
            if x * x + y * y <= (R_ARENA + margin) ** 2:
                pts.append((x, y))
    return np.array(pts)


def max_cover_radius(pts, grid, chunk=4000):
    worst, wp = 0.0, None
    for i in range(0, len(grid), chunk):
        c = grid[i:i + chunk]
        d = np.sqrt(((c[:, None, :] - pts[None, :, :]) ** 2).sum(-1)).min(axis=1)
        k = int(d.argmax())
        if d[k] > worst:
            worst, wp = float(d[k]), c[k].copy()
    return worst, wp


def prune_points(pts, grid, r_target, rng, rounds=3):
    pts = pts.copy()
    for _ in range(rounds):
        for i in rng.permutation(len(pts)):
            if i >= len(pts):
                continue
            cand = np.delete(pts, i, axis=0)
            if max_cover_radius(cand, grid)[0] <= r_target:
                pts = cand
    return pts


def surrounding_bad(pts, grid, k=12, r_surr=R_SURR, chunk=4000):
    """返回不满足环绕条件的网格点(矢量化)。"""
    bad = []
    for i in range(0, len(grid), chunk):
        c = grid[i:i + chunk]
        d = np.sqrt(((c[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        if d.shape[1] < 2:
            bad.append(c)
            continue
        kk = min(k, d.shape[1])
        idx = np.argsort(d, axis=1)[:, :kk]
        dd = np.take_along_axis(d, idx, 1)
        v = np.take_along_axis(pts[None, :, :] - c[:, None, :], idx[:, :, None], 1)
        ang = np.arctan2(v[:, :, 1], v[:, :, 0])
        ang = np.where(dd <= r_surr, ang, np.nan)
        ang.sort(axis=1)
        n_valid = np.sum(~np.isnan(ang), axis=1)
        lo = ang[:, 0]
        last = ang[np.arange(len(ang)), np.maximum(n_valid - 1, 0)]
        gaps = np.diff(ang, axis=1)
        wrap = (lo + 2 * math.pi) - last
        gmax = np.nanmax(np.concatenate([gaps, wrap[:, None]], axis=1), axis=1)
        mask = (n_valid < 2) | (gmax >= math.pi - 1e-9)
        if mask.any():
            bad.append(c[mask])
    return np.vstack(bad) if bad else np.zeros((0, 2))


def route_len(order, start=(0.0, 0.0)):
    s, c = 0.0, np.asarray(start, float)
    for p in order:
        s += float(np.linalg.norm(np.asarray(p, float) - c))
        c = np.asarray(p, float)
    return s


def tsp_order(pts, start=(0.0, 0.0)):
    left = [np.asarray(p, float) for p in pts]
    start = np.asarray(start, float)
    cur, order = start, []
    while left:
        k = min(range(len(left)), key=lambda i: float(((left[i] - cur) ** 2).sum()))
        order.append(left.pop(k))
        cur = order[-1]

    def L(seq):
        s, c = 0.0, start
        for p in seq:
            s += float(np.linalg.norm(p - c))
            c = p
        return s

    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 1, len(order)):
                new = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                if L(new) < L(order) - 1e-9:
                    order, improved = new, True
    return order


def main():
    os.makedirs(RES, exist_ok=True)
    rng = np.random.default_rng(5)
    grid4 = arena_grid(4.0)
    out = {}

    # ---------- 两级侦察点集(风险可控自适应侦察用) ----------
    # level1: 覆盖半径 1400m 的宽松点集(便宜, 先跑一遍); escalation: 补点到 950m 严格覆盖
    lvl = {}
    for target, tag in ((1400.0, "level1"), (1100.0, "level15"), (950.0, "strict")):
        best_pts, best_cost = None, 1e18
        for n in range(3, 13):
            lo, hi = 250.0, 1950.0
            for _ in range(60):
                mid = (lo + hi) / 2
                ring = np.array([[mid * math.cos(2 * math.pi * i / n),
                                  mid * math.sin(2 * math.pi * i / n)] for i in range(n)])
                pts = np.vstack([np.zeros((1, 2)), ring])
                if max_cover_radius(pts, grid4)[0] <= target:
                    hi = mid
                else:
                    lo = mid
            r = hi
            ring = np.array([[r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)]
                             for i in range(n)])
            pts = np.vstack([np.zeros((1, 2)), ring])
            order = tsp_order(pts)
            cost = route_len(order) / 5.0 + 119.0 * len(pts)
            if cost < best_cost:
                best_pts, best_cost = pts, cost
        order = tsp_order(best_pts)
        lvl[tag] = {"points_ordered": [list(map(float, p)) for p in order],
                    "n_points": int(len(order)),
                    "cover_radius_grid4m": max_cover_radius(best_pts, grid4)[0],
                    "route_length_m": route_len(order),
                    "design_cost_s": best_cost}
        print(f"  [cover {tag}] n={len(order)} route={route_len(order):.0f}m "
              f"coverR={max_cover_radius(best_pts, grid4)[0]:.0f}m cost={best_cost:.0f}s", flush=True)

    # escalation: 在 level1 基础上补齐到 950m 严格覆盖(候选只取严格集与粗格点, 保证快速)
    base = np.array(lvl["level1"]["points_ordered"])
    strict = np.array(lvl["strict"]["points_ordered"])
    cand_list = [p for p in strict if min(float(np.linalg.norm(p - q)) for q in base) > 80.0]
    cand = np.array(cand_list) if cand_list else np.zeros((0, 2))
    extra = []
    cur = base.copy()
    while max_cover_radius(cur, grid4)[0] > R_SAFE and len(cand):
        best_i, best_r = 0, 1e18
        for i in range(len(cand)):
            r, _ = max_cover_radius(np.vstack([cur, cand[i]]), grid4)
            if r < best_r:
                best_i, best_r = i, r
        extra.append(cand[best_i])
        cur = np.vstack([cur, cand[best_i]])
        cand = np.delete(cand, best_i, axis=0)
    order_extra = tsp_order(np.array(extra)) if extra else []
    lvl["escalation"] = {"points_ordered": [list(map(float, p)) for p in order_extra],
                         "n_points": len(extra),
                         "cover_radius_after_grid4m": max_cover_radius(cur, grid4)[0]}
    print(f"  [cover escalation] +{len(extra)} 点 → 覆盖半径 "
          f"{max_cover_radius(cur, grid4)[0]:.0f}m", flush=True)
    out["cover_levels"] = lvl

    # 参数族: 中心 + n 个半径 r 的环点; 在覆盖半径<=950 约束下最小化巡线长度
    best3, best_cost = None, 1e18
    for n in range(6, 13):
        lo, hi = 300.0, 1900.0
        for _ in range(60):
            mid = (lo + hi) / 2
            ring = np.array([[mid * math.cos(2 * math.pi * i / n + 0.0),
                              mid * math.sin(2 * math.pi * i / n + 0.0)] for i in range(n)])
            pts = np.vstack([np.zeros((1, 2)), ring])
            if max_cover_radius(pts, grid4)[0] <= R_SAFE:
                hi = mid
            else:
                lo = mid
        r = hi
        ring = np.array([[r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)]
                         for i in range(n)])
        pts = np.vstack([np.zeros((1, 2)), ring])
        order = tsp_order(pts)
        cost = route_len(order) / 5.0 + 119.0 * len(pts)      # 巡线时间 + 全频道侦察代价
        print(f"  [P3 design] n={n} r={r:.0f} coverR={max_cover_radius(pts, grid4)[0]:.1f} "
              f"route={route_len(order):.0f}m cost={cost:.0f}s", flush=True)
        if cost < best_cost:
            best3, best_cost = pts, cost
    pruned = best3
    r1, p1 = max_cover_radius(pruned, grid4)
    r2, _ = max_cover_radius(pruned, arena_grid(1.0))
    order3 = tsp_order(pruned)
    out["problem3"] = {
        "n_points": int(len(pruned)),
        "points_ordered": [list(map(float, p)) for p in order3],
        "cover_radius_grid4m": r1,
        "cover_radius_grid1m": r2,
        "worst_point": list(map(float, p1)),
        "route_length_m": route_len(order3),
        "design_cost_s": route_len(order3) / 5.0 + 119.0 * len(order3),
    }
    print(f"[P3] n={len(pruned)} coverR(4m)={r1:.1f} coverR(1m)={r2:.1f} "
          f"route={route_len(order3):.0f}m", flush=True)

    # ---------- 问题 4: 定向源"环绕条件"代价-收益权衡 ----------
    # 对每个网格点 q: 若 1000m 内侦察点方向最大间隔 g > 180 度, 则存在
    # 比例为 (g-180)/360 的定向方向使该源在所有侦察点都测不到 -> 期望漏检率 f(q)。
    grid10 = arena_grid(10.0)
    grid3 = arena_grid(3.0)
    tradeoff = []
    for s in (1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1645.0):
        for margin in (300.0, 600.0):
            cand4 = hex_lattice(s, margin=margin)
            f10 = miss_probability(cand4, grid10)
            f3 = miss_probability(cand4, grid3)
            order = tsp_order(cand4)
            tradeoff.append({"spacing": s, "margin": margin, "n_points": int(len(cand4)),
                             "route_length_m": route_len(order),
                             "mean_miss_prob": float(f3.mean()),
                             "max_miss_prob": float(f3.max()),
                             "mean_miss_prob_grid10": float(f10.mean()),
                             "points_ordered": [list(map(float, p)) for p in order]})
            print(f"  [P4] s={s:.0f} m={margin:.0f} n={len(cand4)} "
                  f"route={route_len(order):.0f} mean_f={f3.mean():.4f} max_f={f3.max():.3f}",
                  flush=True)
    out["problem4_tradeoff"] = tradeoff

    with open(os.path.join(RES, "cover_points.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved (P3 + P4 tradeoff)")


def miss_probability(pts, grid, k=12, r_surr=R_SURR, chunk=4000):
    """每个网格点的"定向源漏检方向比例" f(q)=(maxgap-180)/360 (无则 0)。"""
    vals = np.zeros(len(grid))
    for i in range(0, len(grid), chunk):
        c = grid[i:i + chunk]
        d = np.sqrt(((c[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        kk = min(k, d.shape[1])
        idx = np.argsort(d, axis=1)[:, :kk]
        dd = np.take_along_axis(d, idx, 1)
        v = np.take_along_axis(pts[None, :, :] - c[:, None, :], idx[:, :, None], 1)
        ang = np.arctan2(v[:, :, 1], v[:, :, 0])
        ang = np.where(dd <= r_surr, ang, np.nan)
        ang.sort(axis=1)
        n_valid = np.sum(~np.isnan(ang), axis=1)
        lo = ang[:, 0]
        last = ang[np.arange(len(ang)), np.maximum(n_valid - 1, 0)]
        gaps = np.diff(ang, axis=1)
        wrap = (lo + 2 * math.pi) - last
        gmax = np.nanmax(np.concatenate([gaps, wrap[:, None]], axis=1), axis=1)
        gmax = np.where(n_valid < 2, 2 * math.pi, gmax)
        vals[i:i + chunk] = np.clip((gmax - math.pi) / (2 * math.pi), 0.0, 1.0)
    return vals


def _deprecated_main():
    chosen, chosen_s = None, None
    for s in (1700.0, 1600.0, 1500.0, 1400.0, 1300.0, 1200.0, 1100.0, 1000.0, 950.0):
        cand4 = hex_lattice(s, margin=250.0)
        bad = surrounding_bad(cand4, arena_grid(10.0))
        print(f"  [P4] spacing={s:.0f} n={len(cand4)} bad(10m)={len(bad)}", flush=True)
        if len(bad) == 0:
            bad_fine = surrounding_bad(cand4, arena_grid(3.0))
            print(f"        verify(3m) bad={len(bad_fine)}", flush=True)
            if len(bad_fine) == 0:
                chosen, chosen_s = cand4, s
                break
    if chosen is None:
        chosen, chosen_s = hex_lattice(950.0, margin=250.0), 950.0
    order4 = tsp_order(chosen)
    out["problem4"] = {
        "spacing": chosen_s, "n_points": int(len(chosen)),
        "points_ordered": [list(map(float, p)) for p in order4],
        "route_length_m": route_len(order4),
        "verify_bad_grid3m": int(len(surrounding_bad(np.array(order4), arena_grid(3.0)))),
    }
    print(f"[P4] chosen spacing={chosen_s:.0f} n={len(chosen)} "
          f"route={route_len(order4):.0f}m", flush=True)
    with open(os.path.join(RES, "cover_points.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

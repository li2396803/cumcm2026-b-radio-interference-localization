"""机器狗搜索-定位-清除策略。

核心思想
 1) 覆盖侦察: 选一组"侦察点", 其 1000m 圆盘并集覆盖目标圆域(问题4 用更密的
    "环绕条件"点集), 在每个侦察点对所有未知频道测量 —— 既发现干扰源, 又构成
    "该频道不存在" 的可证明判据。
 2) 交会定位: 每次测得示向度即得到一条 2° 楔形约束, 与已有约束(以及目标圆域)求交,
    得到候选区域(凸多边形); 区域半径 <= 19.5m 时走到最小包围圆心清除即必定成功。
 3) 追踪式逼近: 剩余区域较大时, 朝候选区域中心前进并做侧向偏移测量(交会角 60°~90°),
    逐步收紧; 小区域时用 20m 覆盖格点做"清除扫描", 用 3s 的 /clear 代替 6s 的检测。
输出统计由 sim.Simulator 汇总。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from geom import (
    clip_polygon, convex_hull, diameter_rotating_calipers_pair, mec, norm, offset_polygon,
    polygon_area, polygon_centroid, point_in_polygon, unit, wedge_halfplanes,
)
from sim import R_ARENA, R_MIN, SPEED, Simulator

R_CLEAR = 20.0
MARGIN = 0.5            # 清除判据留出的安全余量(m)


def disk_polygon(center, radius, n=120):
    c = np.asarray(center, float)
    rr = radius / math.cos(math.pi / n)
    return [c + rr * unit(360.0 * i / n) for i in range(n)]


@dataclass
class Params:
    name: str = "adaptive"
    interleave: bool = True          # True: 侦察与清除插序; False: 先侦察后清除
    use_tracking: bool = True        # 是否用交会追踪逼近
    use_sweep: bool = True           # 是否使用清除扫描
    sweep_max_diam: float = 95.0     # 区域直径小于该值时可做清除扫描
    sweep_spacing: float = 33.0      # 清除扫描格点间距(覆盖半径 19.05m < 20m)
    source_weight: float = 0.85      # 选择目标时对"清除源"的偏好
    lateral_frac: float = 0.30       # 侧向偏移比例
    max_iter_per_source: int = 40


class Robot:
    def __init__(self, sim: Simulator, cover_pts, params: Params, problem=3):
        self.sim = sim
        self.params = params
        self.problem = problem
        self.cover = [np.asarray(p, float) for p in cover_pts]
        self.pending = list(self.cover)
        self.unknown = set(range(1, 21))
        self.tracked: dict[int, list] = {}
        self.cleared: set[int] = set()
        self.proven_empty: set[int] = set()
        self.side: dict[int, float] = {}
        self.n_measure = 0
        self.n_clear = 0
        self.n_home_steps = 0
        self.anomaly = 0
        self.gave_up: list[int] = []
        self.nosig: dict[int, int] = {}
        self.ray: dict[int, tuple] = {}      # ch -> (apex, theta_deg) 最近一次成功示向度
        self.mlog: dict[int, list] = {}      # ch -> [(q, kind, svd), ...] 观测记录
        self.n_filter = 0

    # ------------------------------------------------------------------
    def _dist(self, p):
        return norm(np.asarray(p, float) - self.sim.pos)

    def _scan_unknown(self, at=None):
        """在侦察点测量: 所有未知频道 + 已发现但区域仍较大的频道(顺带获得新示向度)。

        at 给定时, 第一个检测动作完成移动。
        """
        extra = []
        for ch, poly in self.tracked.items():
            if ch in self.cleared:
                continue
            D, _ = diameter_rotating_calipers_pair(poly)
            if D > 80.0:
                extra.append(ch)
        chans = sorted(self.unknown) + extra
        for k, ch in enumerate(chans):
            p = at if (k == 0 and at is not None) else self.sim.pos
            res, svd = self.sim.measure(p, ch)
            self.n_measure += 1
            if res == "direction":
                if ch in self.tracked:
                    self._update_region(ch, self.sim.pos, svd)
                else:
                    self._init_region(ch, svd)
                self.ray[ch] = (self.sim.pos.copy(), svd)
                self.nosig[ch] = 0
            elif res == "no_signal" and ch in self.tracked:
                self._record_nosig(ch, self.sim.pos)
                if self.problem == 4:
                    np2 = self._shadow_filter(ch, self.tracked[ch], prob4=True)
                    if np2 is not None:
                        self.tracked[ch] = np2
            elif res == "near":
                r, _ = self.sim.clear(self.sim.pos, ch)
                self.n_clear += 1
                if r == "success":
                    self.cleared.add(ch)
                    self.unknown.discard(ch)

    def _init_region(self, ch, svd):
        self.mlog.setdefault(ch, []).append((self.sim.pos.copy(), "direction", float(svd)))
        pos = self.sim.pos.copy()
        hps = wedge_halfplanes(pos, svd, 1.0)
        poly = disk_polygon((0, 0), R_ARENA, 360)
        for hp in hps:
            poly = clip_polygon(poly, hp)
            if len(poly) == 0:
                break
        # 在 S 处测到信号 => 源距 S 不超过 1500m
        if len(poly) > 0:
            disk = disk_polygon(pos, 1500.0, 120)
            for hp in _poly_halfplanes(disk):
                poly = clip_polygon(poly, hp)
                if len(poly) == 0:
                    break
        if len(poly) >= 3:
            self.tracked[ch] = _simplify(poly)
            self.unknown.discard(ch)
        else:
            self.anomaly += 1

    def _update_region(self, ch, pos, svd):
        self.mlog.setdefault(ch, []).append((np.asarray(pos, float).copy(), "direction", float(svd)))
        poly = self.tracked.get(ch)
        if poly is None:
            self._init_region(ch, svd)
            return
        poly = [p.copy() for p in poly]
        for hp in wedge_halfplanes(np.asarray(pos, float), svd, 1.0):
            poly = clip_polygon(poly, hp)
            if len(poly) == 0:
                break
        if len(poly) >= 3 and polygon_area(poly) > 1e-6:
            self.tracked[ch] = _simplify(poly)
        else:
            self.anomaly += 1     # 约束矛盾(理论上不会发生)

    # ------------------------------------------------------------------
    def _shadow_filter(self, ch, poly, spacing=15.0, nu=720, prob4=True):
        """用"位置-朝向"联合可行域收缩候选区域。

        观测量约束(全部为保守外近似):
          direction(q, svd):  |p-q| <= 1500 且 |angle(p-q)-svd| <= 1度
          no_signal(q)     :  |p-q| > 1000  或  (q-p).u < 0   (定向源背向)
                             问题3(全向源)时直接使用 |p-q| > 1000
        取可行位置点的凸包(外扩一个格距)作为新的候选区域, 保证仍包含真实源。
        """
        import numpy as np
        from geom import convex_hull, point_in_polygon
        P = np.array(poly, dtype=float)
        lo, hi = P.min(axis=0) - spacing, P.max(axis=0) + spacing
        nx = max(2, int((hi[0] - lo[0]) / spacing) + 1)
        ny = max(2, int((hi[1] - lo[1]) / spacing) + 1)
        if nx * ny > 40000:
            return None
        xs = lo[0] + spacing * np.arange(nx)
        ys = lo[1] + spacing * np.arange(ny)
        GX, GY = np.meshgrid(xs, ys)
        grid = np.stack([GX.ravel(), GY.ravel()], axis=1)
        inside = np.array([point_in_polygon(g, poly) for g in grid])
        pts = grid[inside]
        if len(pts) == 0:
            return None
        U = np.stack([np.cos(2 * np.pi * np.arange(nu) / nu),
                      np.sin(2 * np.pi * np.arange(nu) / nu)], axis=1)   # (nu,2)
        ok_dir = np.ones(len(pts), dtype=bool)
        ok_u = np.ones((len(pts), nu), dtype=bool)
        tol = spacing * 1.5
        use_u = bool(prob4)
        for q, kind, svd in self.mlog.get(ch, []):
            v = pts - q
            d = np.linalg.norm(v, axis=1)
            if kind == "direction":
                ok_dir &= (d <= 1500.0 + tol)
                ang = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
                diff = np.abs((ang - svd + 180.0) % 360.0 - 180.0)
                tol_ang = np.degrees(np.arctan2(tol, np.maximum(d, 1.0))) + 1.0
                ok_dir &= (diff <= tol_ang)
                if use_u:
                    # 源指向测点: (q-p).u >= 0  <=>  vhat.u <= 0
                    vh = v / np.maximum(d, 1e-9)[:, None]
                    ok_u &= (vh @ U.T <= 0.02)
            else:
                if use_u:
                    sel = d <= 1000.0 - tol
                    if sel.any():
                        vh = v[sel] / np.maximum(d[sel], 1e-9)[:, None]
                        ok_u[sel] = ok_u[sel] & (vh @ U.T > -0.02)
                else:
                    ok_dir &= (d > 1000.0 - tol)
        feas = ok_dir & ok_u.any(axis=1)
        if feas.sum() < 3:
            return None
        hull = convex_hull(pts[feas])
        if len(hull) < 3:
            return None
        poly2 = offset_polygon(hull, spacing * 1.5)     # 外扩一个格距, 保持包含真实源
        # 与原区域求交(凸包外扩后仍以原区域为界)
        for hp in _poly_halfplanes(poly):
            poly2 = clip_polygon(poly2, hp)
            if len(poly2) == 0:
                return None
        self.n_filter += 1
        return poly2 if len(poly2) >= 3 else None

    def _record_nosig(self, ch, q):
        self.mlog.setdefault(ch, []).append((np.asarray(q, float).copy(), "no_signal", None))

    # ------------------------------------------------------------------
    def _est_pos(self, ch):
        poly = self.tracked[ch]
        return polygon_centroid(poly)

    def _near_edge(self, ch):
        """区域中离当前位置最近的点(用于估计行程)。"""
        poly = self.tracked[ch]
        P = np.array(poly)
        d = np.linalg.norm(P - self.sim.pos, axis=1)
        return P[int(d.argmin())]

    def _next_measure_point(self, ch, poly):
        # 若有最近示向度射线: 沿射线"螺旋推进"(边前进边横向偏移), 保证始终在定向源前向半平面内
        if self.problem == 4 and ch in self.ray:
            apex, th = self.ray[ch]
            u = unit(th)
            n = np.array([-u[1], u[0]])
            tmax = max(float((np.asarray(p, float) - apex) @ u) for p in poly)
            tmax = max(tmax, 50.0)
            t_now = float((self.sim.pos - apex) @ u)
            t = min(max(t_now + 200.0, 200.0), tmax)
            sgn = self.side.get(ch, 1.0)
            self.side[ch] = -sgn
            lat = 25.0 * sgn if self.nosig.get(ch, 0) == 0 else 12.0 * sgn
            return apex + t * u + lat * n
        # 细长条区域: 沿脊线(直径方向)推进, 避免在质心方向上来回震荡
        D, (a, b) = diameter_rotating_calipers_pair(poly)
        area = polygon_area(poly)
        width = area / D if D > 1e-9 else 0.0
        if width <= 45.0 and D > 80.0:
            u = (b - a) / D
            s = float(np.clip((self.sim.pos - a) @ u, 0.0, D))
            t = s + 250.0
            if t > D - 40.0:
                t = max(D * 0.5, s - 250.0) if s > D * 0.5 else D
            lat = 0.0 if self.params.lateral_frac <= 0 else 12.0
            sgn = self.side.get(ch, 1.0)
            self.side[ch] = -sgn
            n = np.array([-u[1], u[0]])
            return a + float(np.clip(t, 0.0, D)) * u + sgn * lat * n
        c = polygon_centroid(poly)
        v = c - self.sim.pos
        L = norm(v)
        if L < 1e-6:
            u = unit(self.sim.channel * 37.0)
        else:
            u = v / L
        up = np.array([-u[1], u[0]])
        sgn = self.side.get(ch, 1.0)
        self.side[ch] = -sgn
        if L > 700:
            f, lat = 0.55 * L, self.params.lateral_frac * L
        elif L > 150:
            f, lat = 0.45 * L, 0.5 * L
        else:
            f, lat = 0.15 * L, max(60.0, 0.9 * L)
        return self.sim.pos + f * u + sgn * lat * up

    def _clear_sweep(self, ch, poly):
        from geom import cover_lattice_points
        pts = cover_lattice_points(poly, self.params.sweep_spacing, start=self.sim.pos)
        for p in pts:
            r, _ = self.sim.clear(p, ch)
            self.n_clear += 1
            if r == "success":
                return True
        return False

    def _ray_sweep(self, ch):
        """沿最近示向度射线做清除扫掠(横向 ±13m 折线, 清除点间距<=30m)。

        源必在射线两侧 1° 楔形内(横向偏差 <= 0.0175*R <= 26m), 故该扫掠必命中。
        """
        if ch not in self.ray:
            return False
        apex, th = self.ray[ch]
        u = unit(th)
        n = np.array([-u[1], u[0]])
        poly = self.tracked[ch]
        ts = [float((np.asarray(p, float) - apex) @ u) for p in poly]
        t_lo = max(0.0, min(ts) - 20.0)
        t_hi = max(ts) + 30.0
        # 从离当前位置最近的一端开始扫
        t_now = float((self.sim.pos - apex) @ u)
        if abs(t_now - t_lo) > abs(t_now - t_hi):
            t, t_end, step = t_hi, t_lo, -15.0
        else:
            t, t_end, step = t_lo, t_hi, 15.0
        sgn = 1.0
        while (step > 0 and t <= t_end) or (step < 0 and t >= t_end):
            q = apex + t * u + sgn * 13.0 * n
            r, _ = self.sim.clear(q, ch)
            self.n_clear += 1
            if r == "success":
                return True
            t += step
            sgn = -sgn
        return False

    # ------------------------------------------------------------------
    def home_and_clear(self, ch) -> bool:
        """追踪定位并清除指定频道, 返回是否成功。"""
        for _ in range(self.params.max_iter_per_source):
            poly = self.tracked.get(ch)
            if poly is None or len(poly) < 3:
                return False
            c, r = mec(poly)
            D, _ = diameter_rotating_calipers_pair(poly)
            if not self.params.use_tracking:
                # 不做示向度精化: 直接对候选区域做清除扫描
                if D > 1e-9 and self._clear_sweep(ch, poly):
                    return True
                q = self._next_measure_point(ch, poly)
                res, svd = self.sim.measure(q, ch)
                self.n_measure += 1
                self.n_home_steps += 1
                if res == "direction":
                    self._update_region(ch, q, svd)
                elif res == "near":
                    r2, _ = self.sim.clear(q, ch)
                    self.n_clear += 1
                    if r2 == "success":
                        return True
                continue
            if r <= R_CLEAR - MARGIN:
                res, _ = self.sim.clear(c, ch)
                self.n_clear += 1
                if res == "success":
                    return True
                self.anomaly += 1
                return False
            if self.params.use_sweep and D <= self.params.sweep_max_diam and D > 1e-9:
                if self._clear_sweep(ch, poly):
                    return True
                self.anomaly += 1
                # 扫描失败说明区域估计可能已失效: 退回继续检测
            q = self._next_measure_point(ch, poly)
            res, svd = self.sim.measure(q, ch)
            self.n_measure += 1
            self.n_home_steps += 1
            if res == "direction":
                self.nosig[ch] = 0
                self.ray[ch] = (q.copy(), svd)
                self._update_region(ch, q, svd)
            elif res == "near":
                r2, _ = self.sim.clear(q, ch)
                self.n_clear += 1
                if r2 == "success":
                    return True
            else:      # no_signal
                self.nosig[ch] = self.nosig.get(ch, 0) + 1
                self._record_nosig(ch, self.sim.pos)
                if self.problem == 4:
                    np2 = self._shadow_filter(ch, self.tracked[ch], prob4=True)
                    if np2 is not None:
                        self.tracked[ch] = np2
                        poly = self.tracked[ch]
                # (1) 就地清除: 只花 3s, 命中即成功(清除与定向角度无关)
                r3, _ = self.sim.clear(self.sim.pos, ch)
                self.n_clear += 1
                if r3 == "success":
                    return True
                # (2) 定向源背向导致长时间无示向度: 用清除扫描兜底(保证性方法)
                if self.problem == 4:
                    D2, (a2, b2) = diameter_rotating_calipers_pair(poly)
                    width2 = polygon_area(poly) / D2 if D2 > 1e-9 else 0.0
                    if self.nosig[ch] >= 3 and ch in self.ray:
                        if self._ray_sweep(ch):
                            return True
                    elif self.nosig[ch] >= 5 and width2 > 45.0:
                        if self._clear_sweep(ch, poly):
                            return True
        return False

    # ------------------------------------------------------------------
    def run(self) -> dict:
        # 起点即第一个侦察点: 先扫描一次
        self._scan_unknown()
        self.pending = [q for q in self.pending if self._dist(q) > 1e-9]
        while True:
            todo_src = [ch for ch in self.tracked if ch not in self.cleared]
            need_scan = bool(self.unknown) and bool(self.pending)
            if not todo_src and not need_scan:
                break

            options = []
            if todo_src:
                for ch in todo_src:
                    p = self._est_pos(ch) if self.params.interleave else self._near_edge(ch)
                    cost = self._dist(p) * self.params.source_weight
                    poly = self.tracked[ch]
                    D, _ = diameter_rotating_calipers_pair(poly)
                    cost += 0.12 * D + 30.0
                    options.append((cost, "home", ch, p))
            if need_scan:
                P = min(self.pending, key=lambda q: self._dist(q))
                cost = self._dist(P) + 6.0 * len(self.unknown)
                if self.params.interleave:
                    cost *= 1.0
                else:
                    cost *= 0.0      # 非插序: 一律先完成侦察
                options.append((cost, "scan", None, P))
            if not options:
                break
            options.sort(key=lambda o: o[0])
            _, kind, ch, target = options[0]

            if kind == "scan":
                self.pending = [q for q in self.pending if norm(q - target) > 1e-9]
                self._scan_unknown(at=target)     # 移动 + 全未知频道测量
            else:
                ok = self.home_and_clear(ch)
                if ok:
                    self.cleared.add(ch)
                else:
                    # 已有区域但无法清除: 视为已被清除/异常, 避免死循环
                    self.cleared.add(ch)
                    self.gave_up.append(ch)
        return self.finish()

    def finish(self) -> dict:
        sim = self.sim
        # 剩余未知频道: 若所有侦察点已访问, 则可证明这些频道不存在
        proven = set(self.unknown) if not self.pending else set()
        return {
            "cleared": sorted(self.cleared),
            "tracked_channels": sorted(self.tracked),
            "gave_up": sorted(self.gave_up),
            "proven_empty": sorted(proven),
            "unresolved": sorted(self.unknown - proven),
            "n_measure": self.n_measure, "n_clear": self.n_clear,
            "n_home_steps": self.n_home_steps, "anomaly": self.anomaly,
            "stats": sim.stats(),
        }


def _poly_halfplanes(poly):
    """凸多边形各边转为半平面 n.p <= c (逆时针)。"""
    hps = []
    m = len(poly)
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        e = b - a
        n = np.array([e[1], -e[0]])          # 外法向(逆时针多边形)
        nn = norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        hps.append((n, float(n @ a)))
    return hps


def _simplify(poly, tol=0.05):
    from geom import simplify_polygon
    out = simplify_polygon(poly, tol)
    return out


def run_trial(sources, cover_pts, params: Params, problem=3):
    sim = Simulator(sources)
    robot = Robot(sim, cover_pts, params, problem=problem)
    res = robot.run()
    res["sources"] = [{"ch": s.channel, "pos": s.pos.tolist(), "R": s.radius,
                       "kind": s.kind, "dir": s.direction_deg, "cleared": s.cleared}
                      for s in sources]
    return res

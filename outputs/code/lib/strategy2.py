"""v2 策略: 三项改进

改进 1  滚动时域重优化调度(RHO): 每个决策时刻用最新信息重建全部剩余任务(未访问侦察点 +
        已发现未清除源)并重新求解巡线顺序(最近邻 + 2-opt), 只执行第一条腿; 取代 v1 的
        一次性贪心打分。
改进 2  问题4 显式朝向贝叶斯信念: 维护 (位置 p, 朝向 u) 联合假设集与 u 的边缘后验;
        用后验均值把机器狗主动引导到源的"前向侧"取数, 而不是盲目沿射线扫掠;
        仅在信念长期不可用时才动用保证性射线扫掠。
改进 3  风险可控自适应侦察 + R 先验在线学习: 先执行宽松覆盖(level1), 用
        "随机源漏检概率 mu" 与观测到的检测距离反推有效接收半径先验 U(1000,b),
        按期望漏检源个数阈值 theta 决定是否补点升级到 950m 严格覆盖。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from geom import (
    clip_polygon, convex_hull, cover_lattice_points, diameter_rotating_calipers_pair, mec,
    norm, offset_polygon, point_in_polygon, polygon_area, polygon_centroid, unit,
    wedge_halfplanes,
)
from sim import R_ARENA, R_MAX, R_MIN, SPEED
from strategy import MARGIN, R_CLEAR, Params, Robot, _poly_halfplanes, _simplify


@dataclass
class Params2(Params):
    name: str = "v2"
    theta_miss: float = 0.0         # 期望漏检源个数阈值(0=严格; >0 为风险预算, 可提前停止侦察)
    rho: bool = True                # 滚动时域重优化
    use_belief: bool = True         # 问题4 朝向信念
    r_learn: bool = True            # R 先验在线学习
    front_len: float = 350.0        # 前向侧取数点距估计源的距离
    nosig_sweep: int = 4            # 连续无信号多少次后动用保证性射线扫掠
    beta_gain: float = 8000.0       # 风险收益折算为等效行程(m / 个期望漏检源)


# ---------------------------------------------------------------------------
# 改进 3: 漏检风险模型
# ---------------------------------------------------------------------------
class MissModel:
    """给定侦察点集, 估计"随机源未被任何侦察点发现"的概率 mu。

    R ~ U(1000, b), b 由检测到的源的最小检测距离 d_i (满足 R_i >= d_i) 做贝叶斯更新。
    问题3(全向): mu(q) = E_R[ clamp((d_min(q)-1000)/(b-1000)) ]
    问题4(定向): mu(q) = E_u E_R[ clamp((d0(q,u)-1000)/(b-1000)) ],
                 d0 = 位于 u 前向半平面内的最近侦察点距离(无则视为无穷, 即必漏)
    """

    def __init__(self, problem=3, grid_step=25.0, nu=90):
        self.problem = problem
        self.nu = nu
        xs = np.arange(-R_ARENA, R_ARENA + grid_step, grid_step)
        X, Y = np.meshgrid(xs, xs)
        m = (X ** 2 + Y ** 2) <= R_ARENA ** 2
        self.grid = np.stack([X[m], Y[m]], axis=1)
        self.U = np.stack([np.cos(2 * np.pi * np.arange(nu) / nu),
                           np.sin(2 * np.pi * np.arange(nu) / nu)], axis=1)
        self.b_grid = np.arange(1300.0, 1500.1, 10.0)
        self.b_post = None
        self.detected_dists: list[float] = []
        self._cache: dict = {}
        self._ver = 0

    def observe_detected(self, dist: float):
        self.detected_dists.append(float(dist))
        self._update_b()

    def _update_b(self):
        if not self.detected_dists or not np.isfinite(self.detected_dists).all():
            self.b_post = None
            return
        dmax = max(self.detected_dists)
        w = np.ones_like(self.b_grid)
        for d in self.detected_dists:
            w = w * np.clip((self.b_grid - d) / (self.b_grid - R_MIN), 1e-6, 1.0)
        w = np.where(self.b_grid >= max(dmax, R_MIN), w, 0.0)
        w = np.where(self.b_grid <= R_MAX, w, 0.0)
        self.b_post = w / w.sum() if w.sum() > 0 else None
        self._ver += 1
        self._cache.clear()

    def mu(self, pts) -> float:
        pts = np.asarray(pts, float)
        if len(pts) == 0:
            return 1.0
        key = (self._ver, tuple(sorted((round(float(p[0]), 1), round(float(p[1]), 1))
                                       for p in pts)))
        if key in self._cache:
            return self._cache[key]
        if self.b_post is None:
            bs, ws = np.array([R_MAX]), np.array([1.0])
        else:
            bs, ws = self.b_grid, self.b_post
        if self.problem == 3:
            d = np.sqrt(((self.grid[:, None, :] - pts[None, :, :]) ** 2).sum(-1)).min(axis=1)
            frac = np.clip((d[:, None] - R_MIN) / np.maximum(bs[None, :] - R_MIN, 1e-9), 0.0, 1.0)
            val = float((frac * ws[None, :]).sum(axis=1).mean())
            self._cache[key] = val
            return val
        # 定向: 对朝向取平均
        v = self.grid[:, None, :] - pts[None, :, :]            # (g, m, 2) q - P
        dist = np.sqrt((v ** 2).sum(-1))                        # (g, m)
        front = np.einsum("gmc,uc->gmu", v, self.U) >= 0.0      # 侦察点是否在 u 前向半平面
        dd = np.where(front, dist[:, :, None], np.inf).min(axis=1)   # (g, u)
        frac = np.clip((dd[:, :, None] - R_MIN) / np.maximum(bs[None, None, :] - R_MIN, 1e-9),
                       0.0, 1.0)
        val = float((frac * ws[None, None, :]).sum(axis=2).mean())
        self._cache[key] = val
        return val

    def expected_missed(self, pts, n_unresolved: int, n_detected: int, n_total_ch=20,
                        n_max_prior=16, n_min_prior=10) -> float:
        """期望漏检源个数 = E[N|数据] - 已发现数, 用二项似然对 N 做后验。"""
        mu = self.mu(pts)
        ns = np.arange(n_min_prior, n_max_prior + 1)
        logp = []
        for n in ns:
            k = n - n_detected
            if k < 0 or k > n:
                logp.append(-1e9)
                continue
            lp = (k * math.log(max(mu, 1e-12)) + n_detected * math.log(max(1 - mu, 1e-12)))
            lp += math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n_detected + 1)
            logp.append(lp)
        logp = np.array(logp)
        logp -= logp.max()
        p = np.exp(logp)
        p /= p.sum()
        return float((p * (ns - n_detected)).sum())


# ---------------------------------------------------------------------------
# 改进 2: 朝向贝叶斯信念
# ---------------------------------------------------------------------------
class DirBelief:
    """(位置 p, 朝向 u) 联合假设集; 提供 p 的边缘可行域与 u 的边缘后验。"""

    def __init__(self, nu=72):
        self.log: list = []
        self.nu = nu
        self.last = None

    def add(self, q, kind, svd=None):
        self.log.append((np.asarray(q, float).copy(), kind, None if svd is None else float(svd)))

    def evaluate(self, poly, spacing=None):
        if len(poly) < 3 or not self.log:
            return None
        P = np.asarray(poly, float)
        lo, hi = P.min(axis=0), P.max(axis=0)
        if spacing is None:
            spacing = max(14.0, math.sqrt(max((hi - lo).prod(), 1.0) / 1200.0))
        nx = max(2, int((hi[0] - lo[0]) / spacing) + 1)
        ny = max(2, int((hi[1] - lo[1]) / spacing) + 1)
        if nx * ny > 2500:
            return None
        xs = lo[0] + spacing * np.arange(nx)
        ys = lo[1] + spacing * np.arange(ny)
        GX, GY = np.meshgrid(xs, ys)
        grid = np.stack([GX.ravel(), GY.ravel()], axis=1)
        inside = np.array([point_in_polygon(g, poly) for g in grid])
        pts = grid[inside]
        if len(pts) < 3:
            return None
        U = np.stack([np.cos(2 * np.pi * np.arange(self.nu) / self.nu),
                      np.sin(2 * np.pi * np.arange(self.nu) / self.nu)], axis=1)
        ok_dir = np.ones(len(pts), dtype=bool)
        ok_u = np.ones((len(pts), self.nu), dtype=bool)
        tol = spacing * 1.5
        for q, kind, svd in self.log:
            v = pts - q
            d = np.linalg.norm(v, axis=1)
            if kind == "direction":
                ok_dir &= (d <= R_MAX + tol)
                ang = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
                diff = np.abs((ang - svd + 180.0) % 360.0 - 180.0)
                tol_ang = np.degrees(np.arctan2(tol, np.maximum(d, 1.0))) + 1.0
                ok_dir &= (diff <= tol_ang)
                vh = v / np.maximum(d, 1e-9)[:, None]
                ok_u &= (vh @ U.T <= 0.02)          # (q-p).u >= 0
            else:
                sel = d <= R_MIN - tol
                if sel.any():
                    vh = v[sel] / np.maximum(d[sel], 1e-9)[:, None]
                    ok_u[sel] = ok_u[sel] & (vh @ U.T > -0.02)   # (q-p).u < 0
        feas = ok_dir & ok_u.any(axis=1)
        if feas.sum() < 3:
            return None
        w_u = ok_u[feas].sum(axis=0).astype(float)
        if w_u.sum() <= 0:
            return None
        w_u /= w_u.sum()
        c = float((w_u * np.cos(2 * np.pi * np.arange(self.nu) / self.nu)).sum())
        s = float((w_u * np.sin(2 * np.pi * np.arange(self.nu) / self.nu)).sum())
        mean_u = math.degrees(math.atan2(s, c)) % 360.0
        conc = math.hypot(c, s)                       # 0..1, 越大越集中
        hull = convex_hull(pts[feas])
        if len(hull) < 3:
            return None
        region = offset_polygon(hull, spacing * 1.5)
        for hp in _poly_halfplanes(poly):
            region = clip_polygon(region, hp)
            if len(region) == 0:
                return None
        self.last = {"region": region if len(region) >= 3 else None,
                     "mean_u": mean_u, "conc": conc, "u_post": w_u.copy(),
                     "p_mean": pts[feas].mean(axis=0), "spacing": spacing,
                     "n_feasible": int(feas.sum())}
        return self.last


class RobotV2(Robot):
    """滚动时域 + 信念 + 风险可控侦察的机器狗策略。"""

    def __init__(self, sim, plan, params: Params2, problem=3):
        self.plan1 = [np.asarray(p, float) for p in plan["level1"]]
        self.plan_esc = [np.asarray(p, float) for p in plan.get("escalation", [])]
        super().__init__(sim, self.plan1 + self.plan_esc, params, problem=problem)
        self.params: Params2 = params
        self.miss_model = MissModel(problem=problem)
        self.beliefs: dict[int, DirBelief] = {}      # 每个频道独立的朝向信念
        self.escalated = False
        self.scanned: list = []
        self.n_esc_used = 0
        self.scan_set: list = []          # 只记录"已实际访问并测量过"的侦察点

    # ---------------- 观测记录(供风险模型与信念使用) ----------------
    def _init_region(self, ch, svd):
        super()._init_region(ch, svd)
        if self.problem == 4:
            self.beliefs.setdefault(ch, DirBelief()).add(self.sim.pos, "direction", svd)

    def _update_region(self, ch, pos, svd):
        super()._update_region(ch, pos, svd)
        if self.problem == 4:
            self.beliefs.setdefault(ch, DirBelief()).add(pos, "direction", svd)

    def _record_nosig(self, ch, q):
        super()._record_nosig(ch, q)
        if self.problem == 4:
            self.beliefs.setdefault(ch, DirBelief()).add(q, "no_signal")

    def _shadow_filter(self, ch, poly, spacing=15.0, nu=720, prob4=True):
        """v2: 由 DirBelief 统一处理(更快), 这里禁用 v1 的网格滤波。"""
        return None

    # ---------------- 改进 3: 风险可控侦察 ----------------
    def _need_scan(self) -> bool:
        """是否还需要继续侦察(基于"已访问侦察点"的残余漏检风险)。"""
        if not self.unknown or not self.pending:
            return False
        if self.params.theta_miss <= 0.0:
            return True                     # 严格模式: 执行完整个侦察计划(保证性覆盖)
        em = self.miss_model.expected_missed(self.scan_set, len(self.unknown), len(self.cleared))
        return bool(em > self.params.theta_miss)

    def _maybe_escalate(self):
        """风险驱动逐个补点: 每次只加"单位行程收益最大"的一个升级点。"""
        if not self.unknown:
            return
        if any(all(norm(np.asarray(q, float) - s) > 1e-9 for s in self.scanned)
               for q in self.pending):
            return                                  # 已有待执行侦察点
        em0 = self.miss_model.expected_missed(self.scan_set, len(self.unknown), len(self.cleared))
        if em0 <= self.params.theta_miss:
            return
        cand = [q for q in self.plan_esc if all(norm(q - s) > 1e-9 for s in self.scanned)]
        if not cand:
            return
        best, best_gain = None, -1.0
        for q in cand:
            em1 = self.miss_model.expected_missed(list(self.scan_set) + [q],
                                                  len(self.unknown), len(self.cleared))
            gain = (em0 - em1) / max(self._dist(q), 50.0)
            if gain > best_gain:
                best, best_gain = q, gain
        self.pending = list(self.pending) + [np.asarray(best, float)]
        self.escalated = True
        self.n_esc_used += 1

    # ---------------- 改进 1: 滚动时域重优化 ----------------
    def _jobs(self):
        jobs = []
        for ch in self.tracked:
            if ch in self.cleared:
                continue
            poly = self.tracked[ch]
            D, _ = diameter_rotating_calipers_pair(poly)
            c = polygon_centroid(poly)
            jobs.append(("clear", ch, np.asarray(c, float), self._dist(c) + 0.15 * D + 30.0))
        if self._need_scan():
            for P in self.pending:
                jobs.append(("scan", None, np.asarray(P, float),
                             self._dist(P) + 6.0 * len(self.unknown)))
        return jobs

    def _plan_next(self, jobs):
        """把当前点与全部任务目标点做最近邻+2-opt, 返回第一条腿对应的任务。"""
        start = self.sim.pos.copy()
        nodes = [j[2] for j in jobs]
        if len(nodes) == 1:
            return jobs[0]
        # 最近邻
        left = list(range(len(nodes)))
        order, cur = [], start
        while left:
            k = min(left, key=lambda i: norm(nodes[i] - cur))
            order.append(k)
            left.remove(k)
            cur = nodes[k]

        def path_len(seq):
            s, c = 0.0, start
            for i in seq:
                s += norm(nodes[i] - c)
                c = nodes[i]
            return s

        improved = True
        while improved:
            improved = False
            for i in range(len(order) - 1):
                for j in range(i + 1, len(order)):
                    new = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                    if path_len(new) < path_len(order) - 1e-9:
                        order, improved = new, True
        # 交会角/信息价值修正: 同等距离时优先"清除"任务
        first = order[0]
        cand = [i for i in range(len(jobs)) if norm(nodes[i] - nodes[first]) < 1e-6]
        cand.sort(key=lambda i: 0 if jobs[i][0] == "clear" else 1)
        return jobs[cand[0]]

    # ---------------- 侦察 ----------------
    def _do_scan(self, P):
        self.pending = [q for q in self.pending if norm(q - P) > 1e-9]
        self.scanned.append(np.asarray(P, float))
        self._scan_unknown(at=P)
        if all(norm(np.asarray(s, float) - P) > 1e-9 for s in self.scan_set):
            self.scan_set.append(np.asarray(P, float))
        # 用"检测点到候选区域最近点/最远点"的距离更新该频道的 R 区间, 再更新群体先验
        self.miss_model.detected_dists = self._detected_dists()
        self.miss_model._update_b()

    def _detected_dists(self):
        """每个已发现源给出一条 R 的下界估计(区域最近点距离), 作为 b 的支撑约束。"""
        out = []
        for ch in self.tracked:
            if ch in self.cleared:
                continue
            poly = self.tracked[ch]
            best = 0.0
            for s in self.scanned:
                dmin = min(norm(np.asarray(p, float) - np.asarray(s, float)) for p in poly)
                best = max(best, dmin)
            d = best
            out.append(float(d))
        return out if out else [R_MIN]

    # ---------------- 清除(含 v1 的清除扫描与射线扫掠兜底) ----------------
    def _do_clear(self, ch):
        ok = self.home_and_clear(ch)
        if not ok:
            # 最后兜底: 对整个候选区域做 20m 覆盖格点清除扫描(保证性方法)
            poly = self.tracked.get(ch)
            if poly is not None and len(poly) >= 3:
                D, _ = diameter_rotating_calipers_pair(poly)
                n_est = polygon_area(poly) / (0.433 * self.params.sweep_spacing ** 2)
                if n_est <= 900:                     # 规模可控才执行(避免时间爆炸)
                    ok = self._clear_sweep(ch, poly)
        self.cleared.add(ch)
        if not ok:
            self.gave_up.append(ch)

    # ---------------- 问题4: 用信念引导到前向侧 ----------------
    def _next_measure_point(self, ch, poly):
        if (self.problem == 4 and self.params.use_belief and ch in self.beliefs
                and self.nosig.get(ch, 0) > 0):
            b = self.beliefs[ch].evaluate(poly)
            if b is not None:
                if b["region"] is not None and polygon_area(b["region"]) < polygon_area(poly):
                    self.tracked[ch] = _simplify(b["region"], 0.05)
                    poly = self.tracked[ch]
                if b["conc"] > 0.15:
                    u = unit(b["mean_u"])
                    goal = np.asarray(b["p_mean"], float) + self.params.front_len * u
                    v = goal - self.sim.pos
                    L = norm(v)
                    if L > 1e-6:
                        uu = v / L
                        n = np.array([-uu[1], uu[0]])
                        sgn = self.side.get(ch, 1.0)
                        self.side[ch] = -sgn
                        step = min(L, max(120.0, 0.7 * L))
                        return self.sim.pos + step * uu + sgn * 0.18 * step * n
        return super()._next_measure_point(ch, poly)

    # ---------------- 主循环 ----------------
    def run(self):
        self._scan_unknown()
        self.scanned.append(self.sim.pos.copy())
        self.scan_set.append(self.sim.pos.copy())    # 起点(0,0)已全频道测量
        self.pending = [q for q in self.pending if self._dist(q) > 1e-9]
        guard = 0
        while guard < 4000:
            guard += 1
            self._maybe_escalate()
            jobs = self._jobs()
            if not jobs:
                break
            kind, ch, target, _ = self._plan_next(jobs)
            if kind == "scan":
                self._do_scan(target)
            else:
                self._do_clear(ch)
        return self.finish()


def run_trial_v2(sources, plan, params: Params2, problem=3):
    from sim import Simulator
    sim = Simulator(sources)
    robot = RobotV2(sim, plan, params, problem=problem)
    res = robot.run()
    res["sources"] = [{"ch": s.channel, "pos": s.pos.tolist(), "R": s.radius,
                       "kind": s.kind, "dir": s.direction_deg, "cleared": s.cleared}
                      for s in sources]
    res["escalated"] = robot.escalated
    res["n_esc_points"] = robot.n_esc_used
    res["scan_points"] = len(robot.scanned)
    return res

"""无线电干扰源环境模拟器(按赛题附录2 / 附件2 规则实现, 用于本地演练与统计检验)。

与官方模拟器的一致性:
  * 目标区域: 半径 1800m 圆域, 原点为圆心; 机器狗初始位置 (0,0), 初始频道 1
  * 干扰源: 频道互异(1..20), 位置与有效接收半径(1000..1500m)未知
  * 全向源: 距离<=R 即可接收; 定向源: 距离<=R 且位于定向方向 ±90° 内
  * 示向度误差: 同一地点固定(确定性伪随机场), 全局落在 ±1°
  * 计时: 移动 |Δp|/5 s; 换频道 1 s; 检测 5 s; 清除 未发现 3 s / 成功 5 s
  * /clear 不换频道、不受定向覆盖角限制; 5m 内检测返回 near; 20m 内清除成功
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from geom import ang_deg, norm

R_ARENA = 1800.0
SPEED = 5.0
T_MEASURE = 5.0
T_SWITCH = 1.0
T_CLEAR_SUCCESS = 5.0
T_CLEAR_FAIL = 3.0
R_NEAR = 5.0
R_CLEAR = 20.0
R_MIN = 1000.0
R_MAX = 1500.0
V_LIMIT = 360000.0


def _mix(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def env_error(p, channel: int) -> float:
    """固定电磁环境下的示向度误差(度), 同一地点同一频道恒定, 落在 [-1,1]。"""
    qx = int(round(p[0] * 0.2))          # 5m 网格
    qy = int(round(p[1] * 0.2))
    h = _mix((qx & 0xFFFFF) | ((qy & 0xFFFFF) << 20) | (channel << 44))
    return (h / 2 ** 63) - 1.0


@dataclass
class Source:
    channel: int
    pos: np.ndarray
    radius: float
    kind: str = "omni"                 # omni | dir
    direction_deg: float | None = None
    cleared: bool = False

    def covers(self, p) -> bool:
        d = norm(np.asarray(p, float) - self.pos)
        if d > self.radius:
            return False
        if self.kind == "dir":
            u = np.array([math.cos(math.radians(self.direction_deg)),
                          math.sin(math.radians(self.direction_deg))])
            if float((np.asarray(p, float) - self.pos) @ u) < 0.0:
                return False
        return True

    def dist(self, p) -> float:
        return norm(np.asarray(p, float) - self.pos)


@dataclass
class Simulator:
    sources: list
    t: float = 0.0
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    channel: int = 1
    n_requests: int = 0
    log: list = field(default_factory=list)
    cleared: list = field(default_factory=list)
    total_move_m: float = 0.0
    t_move: float = 0.0
    t_switch: float = 0.0
    t_measure: float = 0.0
    t_clear: float = 0.0

    # ---------------- 基础动作 ----------------
    def _move(self, p):
        p = np.asarray(p, float)
        d = norm(p - self.pos)
        dt = d / SPEED
        self.t += dt
        self.total_move_m += d
        self.t_move += dt
        self.pos = p

    def measure(self, pos, channel: int):
        """返回 (result, svd_or_None)。"""
        self.n_requests += 1
        self._move(pos)
        dt = T_MEASURE
        if channel != self.channel:
            dt += T_SWITCH
            self.t_switch += T_SWITCH
            self.channel = channel
        self.t += dt
        self.t_measure += T_MEASURE

        src = None
        for s in self.sources:
            if s.channel == channel and not s.cleared:
                src = s
                break
        if src is None:
            return "no_signal", None
        if not src.covers(self.pos):
            return "no_signal", None
        d = src.dist(self.pos)
        if d <= R_NEAR:
            return "near", None
        svd = (ang_deg(src.pos - self.pos) + env_error(self.pos, channel)) % 360.0
        return "direction", svd

    def clear(self, pos, channel: int):
        self.n_requests += 1
        self._move(pos)
        src = None
        for s in self.sources:
            if s.channel == channel and not s.cleared:
                src = s
                break
        if src is not None and src.dist(self.pos) <= R_CLEAR + 1e-9:
            self.t += T_CLEAR_SUCCESS
            self.t_clear += T_CLEAR_SUCCESS
            src.cleared = True
            self.cleared.append(channel)
            return "success", self.t
        self.t += T_CLEAR_FAIL
        self.t_clear += T_CLEAR_FAIL
        return "no_target_in_range", self.t

    # ---------------- 统计 ----------------
    def stats(self) -> dict:
        n = len(self.sources)
        nc = len(self.cleared)
        return {
            "n_sources": n, "n_cleared": nc,
            "ratio": nc / n if n else 0.0,
            "total_time_s": self.t,
            "avg_time_s": self.t / nc if nc else None,
            "n_requests": self.n_requests,
            "move_m": self.total_move_m,
            "t_move": self.t_move, "t_switch": self.t_switch,
            "t_measure": self.t_measure, "t_clear": self.t_clear,
        }


def generate_case(rng, problem=3, n_sources=None):
    """随机生成一局案例。"""
    if n_sources is None:
        n_sources = int(rng.integers(10, 17))
    channels = rng.choice(np.arange(1, 21), size=n_sources, replace=False)
    srcs = []
    for ch in channels:
        r = R_ARENA * math.sqrt(rng.random())
        a = rng.random() * 2 * math.pi
        pos = np.array([r * math.cos(a), r * math.sin(a)])
        radius = R_MIN + (R_MAX - R_MIN) * rng.random()
        if problem == 3:
            srcs.append(Source(int(ch), pos, radius, "omni", None))
        else:
            kind = "dir" if rng.random() < 0.5 else "omni"
            direction = None if kind == "omni" else float(rng.random() * 360.0)
            srcs.append(Source(int(ch), pos, radius, kind, direction))
    return srcs

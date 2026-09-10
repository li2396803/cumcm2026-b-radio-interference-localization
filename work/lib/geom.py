"""几何与定位区域算法库 (CUMCM2026 B 题)

约定:
  角度单位为度, 逆时针为正, x 轴正向为 0 度.
  半平面统一写成  n . p <= c   (n 为单位外法向).
  凸多边形用顶点列表表示 (逆时针, 首尾不重复).
"""
from __future__ import annotations

import math
import numpy as np

EPS = 1e-12


# ----------------------------------------------------------------------------
# 基本向量工具
# ----------------------------------------------------------------------------
def unit(deg: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([math.cos(r), math.sin(r)])


def cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[0] + a[1] * b[1])


def norm(a: np.ndarray) -> float:
    return float(math.hypot(a[0], a[1]))


def ang_deg(v: np.ndarray) -> float:
    """向量方位角, [0,360)."""
    return math.degrees(math.atan2(v[1], v[0])) % 360.0


def ang_diff(a: float, b: float) -> float:
    """两方位角之差, 归一化到 (-180,180]."""
    d = (a - b) % 360.0
    return d - 360.0 if d > 180.0 else d


# ----------------------------------------------------------------------------
# 半平面 / 楔形 (定位扇区)
# ----------------------------------------------------------------------------
def wedge_halfplanes(S, theta_deg: float, eps_deg: float):
    """示向度 theta 误差 +-eps 对应的楔形区域(以 S 为顶点)。

    返回两个半平面 (n, c), 满足 n.p <= c 的点即为楔形内部。
    楔形 = {p : angle(p-S) 在 [theta-eps, theta+eps] 内}
    """
    hps = []
    for sgn in (+1.0, -1.0):                      # 两条边界射线方向
        d = unit(theta_deg + sgn * eps_deg)
        n = np.array([-d[1], d[0]])               # n.v = cross(d, v)
        if sgn > 0:                               # cross(d+, p-S) <= 0
            hps.append((n, dot(n, S)))
        else:                                     # cross(d-, p-S) >= 0
            hps.append((-n, -dot(n, S)))
    return hps


def hp_feasible_point(hps):
    """求半平面组的一个内点(用极小化最大违反量的方式), 失败返回 None."""
    n = len(hps)
    A = np.array([h[0] for h in hps])
    b = np.array([h[1] for h in hps])
    # 最小二乘 + 迭代投影(CG on 简易 QP): 最小化 ||max(0, A x - b)||^2
    x = np.zeros(2)
    for _ in range(200):
        v = A @ x - b
        act = v > 0
        if not act.any():
            return x
        Aa = A[act]
        g = 2.0 * Aa.T @ (Aa @ x - b[act])
        step = 1.0 / (2.0 * len(Aa) + 1e-9)
        x = x - step * g
    v = A @ x - b
    return x if (v <= 1e-7).all() else None


def is_bounded(hps) -> bool:
    """半平面组 {n_i.p <= c_i} 是否有界。

    判据: 0 落在 {n_i} 凸包内部 <=> {n_i} 不被任何过原点的闭半平面包含
          <=> 法向方位的最大间隔 < 180 度。
    """
    angs = sorted(math.degrees(math.atan2(h[0][1], h[0][0])) % 360.0 for h in hps)
    if len(angs) < 2:
        return False
    gaps = [angs[i + 1] - angs[i] for i in range(len(angs) - 1)]
    gaps.append(angs[0] + 360.0 - angs[-1])
    return max(gaps) < 180.0 - 1e-9


def clip_polygon(poly, hp):
    """Sutherland-Hodgman: 用半平面 n.p <= c 裁剪凸多边形."""
    if not len(poly):
        return poly
    n, c = hp
    out = []
    m = len(poly)
    for i in range(m):
        a = poly[i]
        b = poly[(i + 1) % m]
        va = dot(n, a) - c
        vb = dot(n, b) - c
        if va <= 1e-12:
            out.append(a)
        if (va < -1e-12 < vb) or (vb < -1e-12 < va):
            t = va / (va - vb)
            out.append(a + t * (b - a))
    return dedup(out)


def dedup(poly, tol=1e-9):
    if len(poly) < 2:
        return poly
    out = []
    for p in poly:
        if not out or norm(p - out[-1]) > tol:
            out.append(p)
    if len(out) > 1 and norm(out[0] - out[-1]) <= tol:
        out.pop()
    return out


def polygon_area(poly) -> float:
    m = len(poly)
    if m < 3:
        return 0.0
    s = 0.0
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        s += a[0] * b[1] - b[0] * a[1]
    return abs(s) / 2.0


def polygon_centroid(poly) -> np.ndarray:
    poly = [np.asarray(p, dtype=float) for p in poly]
    m = len(poly)
    if m == 0:
        return np.zeros(2)
    if m == 1:
        return poly[0].copy()
    if m == 2:
        return (poly[0] + poly[1]) / 2.0
    a = 0.0
    cx = cy = 0.0
    for i in range(m):
        p, q = poly[i], poly[(i + 1) % m]
        cr = p[0] * q[1] - q[0] * p[1]
        a += cr
        cx += (p[0] + q[0]) * cr
        cy += (p[1] + q[1]) * cr
    if abs(a) < 1e-12:
        return np.mean(np.array(poly), axis=0)
    return np.array([cx / (3.0 * a), cy / (3.0 * a)])


def offset_polygon(poly, d):
    """凸多边形向外偏移 d (Minkowski 和, 逆时针输入)。"""
    P = [np.asarray(p, dtype=float) for p in poly]
    m = len(P)
    if m < 3:
        return P
    lines = []
    for i in range(m):
        a, b = P[i], P[(i + 1) % m]
        e = b - a
        L = norm(e)
        if L < 1e-12:
            continue
        n = np.array([e[1], -e[0]]) / L          # 逆时针多边形的外法向
        lines.append((a + d * n, b + d * n))
    out = []
    for i in range(len(lines)):
        p1, p2 = lines[i]
        q1, q2 = lines[(i - 1) % len(lines)]
        r, s = p2 - p1, q2 - q1
        den = cross(r, s)
        if abs(den) < 1e-12:
            out.append(p1)
            continue
        t = cross(q1 - p1, s) / den
        out.append(p1 + t * r)
    return dedup(out)


def polygon_perimeter(poly) -> float:
    m = len(poly)
    if m < 2:
        return 0.0
    return float(sum(norm(poly[(i + 1) % m] - poly[i]) for i in range(m)))


def convex_hull(pts):
    pts = sorted({(round(float(p[0]), 9), round(float(p[1]), 9)) for p in pts})
    if len(pts) <= 2:
        return [np.array(p) for p in pts]

    def half(points):
        st = []
        for p in points:
            while len(st) >= 2 and cross(st[-1] - st[-2], np.array(p) - st[-1]) <= 0:
                st.pop()
            st.append(np.array(p))
        return st

    lo = half(pts)
    up = half(pts[::-1])
    return dedup(lo[:-1] + up[:-1])


_CIRC_CACHE = {}


def cached_circ_polygon(r: float, n: int = 360):
    key = (round(float(r), 6), int(n))
    if key not in _CIRC_CACHE:
        _CIRC_CACHE[key] = circ_polygon(key[0], key[1])
    return [p.copy() for p in _CIRC_CACHE[key]]


def circ_polygon(r: float, n: int = 360, center=(0.0, 0.0), circumscribe=True):
    """正 n 边形逼近圆。circumscribe=True 为外切(半径放大 1/cos(pi/n)),
    保证交会区域是对真实区域的外近似(误差 <= r*(sec-1) ~ r*pi^2/(2n^2))."""
    c = np.array(center, dtype=float)
    rr = r / math.cos(math.pi / n) if circumscribe else r
    return [c + rr * unit(360.0 * i / n) for i in range(n)]


# ----------------------------------------------------------------------------
# 定位区域
# ----------------------------------------------------------------------------
def localization_polygon(obs, arena=None, box=4.0e6):
    """上交会定位区域 = 各楔形交集 [∩ 目标区域]。

    obs: [(S, theta, eps), ...]
    arena: 若给定 (半径或顶点列表), 用其截断; 否则用大盒子截断。
    返回 (poly, bounded) ; 无解返回 ([], True/False)。
    """
    hps = []
    for S, th, eps in obs:
        hps.extend(wedge_halfplanes(np.asarray(S, dtype=float), th, eps))
    bounded_geom = is_bounded(hps)

    if isinstance(arena, (int, float)):
        seed = cached_circ_polygon(float(arena), 360)
    elif arena is None:
        b = box
        seed = [np.array([-b, -b]), np.array([b, -b]), np.array([b, b]), np.array([-b, b])]
    else:
        seed = [np.asarray(p, dtype=float) for p in arena]

    poly = [np.asarray(p, dtype=float) for p in seed]
    for hp in hps:
        poly = clip_polygon(poly, hp)
        if len(poly) == 0:
            return [], bounded_geom
    # 退化检查: 面积过小视为空
    if len(poly) >= 3 and polygon_area(poly) < 1e-9:
        return [], bounded_geom
    if len(poly) > 12:
        poly = simplify_polygon(poly, tol=0.02)
    return poly, bounded_geom


# ----------------------------------------------------------------------------
# 直径: 三种独立算法
# ----------------------------------------------------------------------------
def diameter_bruteforce(poly) -> float:
    """方法 A: 顶点两两枚举 O(m^2)。凸集直径必在顶点对取得。"""
    m = len(poly)
    best = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            best = max(best, norm(poly[i] - poly[j]))
    return best


def diameter_pair_bruteforce(poly):
    m = len(poly)
    best, bi, bj = 0.0, 0, 0
    for i in range(m):
        for j in range(i + 1, m):
            d = norm(poly[i] - poly[j])
            if d > best:
                best, bi, bj = d, i, j
    return best, (poly[bi] if m else None), (poly[bj] if m else None)


def diameter_rotating_calipers_pair(poly):
    """旋转卡壳返回 (直径, 端点对)。"""
    pts = convex_hull(poly)
    m = len(pts)
    if m < 2:
        return 0.0, (None, None)
    if m == 2:
        return norm(pts[1] - pts[0]), (pts[0], pts[1])

    def area2(a, b, c):
        return abs(cross(b - a, c - a))

    j = 1
    best, bp = 0.0, (pts[0], pts[1])
    for i in range(m):
        ni = (i + 1) % m
        while True:
            nj = (j + 1) % m
            if area2(pts[i], pts[ni], pts[nj]) > area2(pts[i], pts[ni], pts[j]) + 1e-15:
                j = nj
            else:
                break
        for cand in (pts[i], pts[ni]):
            d = norm(cand - pts[j])
            if d > best:
                best, bp = d, (cand, pts[j])
    return best, bp


def diameter_rotating_calipers(poly) -> float:
    """方法 B: 旋转卡壳, O(m)。返回凸多边形直径。"""
    return diameter_rotating_calipers_pair(poly)[0]


def diameter_support_scan(poly, extra_angles=720) -> float:
    """方法 C: 支撑函数扫描 D = max_phi (h(phi) + h(phi+pi))。

    对多边形, 最大值必在边的外法向处取得; 这里扫描边法向 + 均匀角, 得到上界意义下
    的精确值(边法向已覆盖所有候选)。
    """
    pts = convex_hull(poly)
    if len(pts) < 2:
        return 0.0
    angs = []
    m = len(pts)
    for i in range(m):
        e = pts[(i + 1) % m] - pts[i]
        angs.append(math.atan2(e[1], e[0]) + math.pi / 2.0)
        angs.append(math.atan2(e[1], e[0]) - math.pi / 2.0)
    angs += list(np.linspace(0, 2 * math.pi, extra_angles, endpoint=False))
    P = np.array(pts)
    best = 0.0
    for a in angs:
        n = np.array([math.cos(a), math.sin(a)])
        h1 = float((P @ n).max())
        h2 = float((P @ (-n)).max())
        best = max(best, h1 + h2)
    return best


# ----------------------------------------------------------------------------
# 最小包围圆 (MEC)
# ----------------------------------------------------------------------------
def mec_bruteforce(pts):
    """精确最小包围圆: 枚举 1/2/3 点定圆. 返回 (center, radius)。"""
    P = [np.asarray(p, dtype=float) for p in pts]
    n = len(P)
    if n == 0:
        return np.zeros(2), 0.0
    if n == 1:
        return P[0].copy(), 0.0

    def ok(c, r):
        return all(norm(p - c) <= r + 1e-9 for p in P)

    best_c, best_r = None, float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            c = (P[i] + P[j]) / 2.0
            r = norm(P[i] - P[j]) / 2.0
            if r < best_r - 1e-12 and ok(c, r):
                best_c, best_r = c, r
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c3 = P[i], P[j], P[k]
                d = 2.0 * (a[0] * (b[1] - c3[1]) + b[0] * (c3[1] - a[1]) + c3[0] * (a[1] - b[1]))
                if abs(d) < 1e-12:
                    continue
                ux = ((a @ a) * (b[1] - c3[1]) + (b @ b) * (c3[1] - a[1]) + (c3 @ c3) * (a[1] - b[1])) / d
                uy = ((a @ a) * (c3[0] - b[0]) + (b @ b) * (a[0] - c3[0]) + (c3 @ c3) * (b[0] - a[0])) / d
                c = np.array([ux, uy])
                r = norm(a - c)
                if r < best_r - 1e-12 and ok(c, r):
                    best_c, best_r = c, r
    return best_c, best_r


def mec_minidisk(poly, rng=None):
    """最小包围圆: 随机增量算法 (Welzl 等价形式), 期望 O(n)。

    返回 (center, radius)。结果对全部顶点精确(浮点容差 1e-9)。
    """
    pts = [np.asarray(p, dtype=float) for p in poly]
    if not pts:
        return np.zeros(2), 0.0
    rng = rng or np.random.default_rng(12345)
    idx = rng.permutation(len(pts))
    P = [pts[i] for i in idx]
    c = P[0].copy()
    r = 0.0
    for i in range(1, len(P)):
        p = P[i]
        if norm(p - c) <= r + 1e-9:
            continue
        c, r = p.copy(), 0.0
        for j in range(i):
            q = P[j]
            if norm(q - c) <= r + 1e-9:
                continue
            c = (p + q) / 2.0
            r = norm(p - q) / 2.0
            for k in range(j):
                s = P[k]
                if norm(s - c) <= r + 1e-9:
                    continue
                c, r = circumcircle(p, q, s)
    # 数值兜底
    for _ in range(4):
        bad = [p for p in pts if norm(p - c) > r + 1e-7]
        if not bad:
            break
        r = max(norm(p - c) for p in pts)
    return c, r


def circumcircle(a, b, c):
    d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-14:
        return mec_bruteforce([a, b, c])
    ux = ((a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])) / d
    uy = ((a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])) / d
    ct = np.array([ux, uy])
    return ct, norm(a - ct)


def mec(poly, sample_cap=64, rng=None):
    """凸多边形最小包围圆 (顶点集与多边形等价)。"""
    return mec_minidisk(poly, rng=rng)


def simplify_polygon(poly, tol=0.02):
    """单趟贪心抽稀: 删除到"保留的前一点-后一点"连线距离小于 tol 的顶点。

    O(n)。用于压缩平滑圆弧上的大量采样点。凸性可能有极小破坏, 调用方按需再取凸包。
    """
    m = len(poly)
    if m <= 4:
        return [np.asarray(p, dtype=float) for p in poly]
    keep = [0]
    for i in range(1, m - 1):
        a = poly[keep[-1]]
        b = poly[i]
        c = poly[i + 1]
        e = c - a
        L = norm(e)
        dist = norm(b - a) if L < 1e-12 else abs(cross(e, b - a)) / L
        if dist >= tol:
            keep.append(i)
    keep.append(m - 1)
    out = [np.asarray(poly[i], dtype=float) for i in keep]
    return dedup(out)


def diameter_disk_covers(poly, pair) -> bool:
    """以直径对 [a,b] 为直径的圆是否覆盖 poly。

    判据: 对凸多边形, max_p (p-a).(p-b) <= 0, 只需检查顶点。
    使用相对容差(几何尺度 ~1e3 m, 点积量级 ~1e6 m^2)。
    """
    a, b = pair
    if a is None or b is None:
        return True
    scale = max(norm(a - b) ** 2, 1.0)
    return all(dot(p - a, p - b) <= 1e-9 * scale for p in poly)


# ----------------------------------------------------------------------------
# 采样与蒙特卡洛
# ----------------------------------------------------------------------------
def sample_in_polygon(poly, n, rng):
    """按面积在凸多边形内均匀采样 (三角形扇)。"""
    if len(poly) < 3:
        return np.zeros((0, 2))
    P = np.array(poly, dtype=float)
    tris, areas = [], []
    for i in range(1, len(poly) - 1):
        tris.append((P[0], P[i], P[i + 1]))
        areas.append(abs(cross(P[i] - P[0], P[i + 1] - P[0])) / 2.0)
    areas = np.array(areas)
    tot = areas.sum()
    if tot <= 0:
        return np.zeros((0, 2))
    idx = rng.choice(len(tris), size=n, p=areas / tot)
    out = np.zeros((n, 2))
    for k in range(n):
        a, b, c = tris[idx[k]]
        u, v = rng.random(), rng.random()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        out[k] = a + u * (b - a) + v * (c - a)
    return out


def cover_lattice_points(poly, spacing, start=None, shrink=1.0):
    """用三角格点覆盖多边形(覆盖半径 spacing/sqrt(3)) 的候选清除点, 最近邻排序。"""
    if len(poly) < 3:
        return []
    P = np.array(poly, dtype=float)
    lo, hi = P.min(axis=0), P.max(axis=0)
    dy = spacing * math.sqrt(3) / 2.0
    rows = int((hi[1] - lo[1]) / dy) + 2
    pts = []
    from matplotlib.path import Path  # 仅用于点是否在多边形内
    path = Path(P)
    for r in range(rows):
        y = lo[1] + r * dy
        x0 = lo[0] + (spacing / 2.0 if r % 2 else 0.0)
        cols = int((hi[0] - x0) / spacing) + 2
        for c2 in range(cols):
            x = x0 + c2 * spacing
            if path.contains_point((x, y)):
                pts.append(np.array([x, y]))
    if not pts:
        pts = [polygon_centroid(poly)]
    # 最近邻排序
    cur = np.asarray(start, dtype=float) if start is not None else polygon_centroid(poly)
    order, left = [], list(pts)
    while left:
        k = min(range(len(left)), key=lambda i: norm(left[i] - cur))
        order.append(left.pop(k))
        cur = order[-1]
    return order


def point_in_polygon(p, poly) -> bool:
    ok = True
    m = len(poly)
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        if cross(b - a, np.asarray(p, float) - a) < -1e-9:
            ok = False
            break
    return ok

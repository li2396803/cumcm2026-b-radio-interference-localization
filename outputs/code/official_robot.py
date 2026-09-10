"""官方模拟器机器狗程序(HTTP+JSON 接口实现)。

用法:
    python official_robot.py --robot-id <参赛队号> [--problem 3|4] [--host 127.0.0.1:2026]

说明:
 * 与官方模拟器通信采用 4 条指令: /enter /measure /clear /exit, 严格串行、逐次等待响应。
 * 每个新动作使用新的 request_id; 网络超时重试时复用原 request_id 与原请求内容。
 * 同时检查 HTTP 状态码与 accepted 字段; 使用 /enter 返回的 remaining_real_duration_s
    控制现实运行时间, 临近截止时主动 /exit。
 * 策略与本地演练完全一致: 覆盖侦察 + 交会定位 + 追踪逼近 + 清除扫描。
 * 全过程指令与响应写入 local_log_<robot>.jsonl 以便自查(模拟器日志用于支撑材料)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import numpy as np  # noqa: E402

from strategy import Params, Robot  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


class HttpSim:
    """把官方 HTTP 接口包装成本地策略类需要的接口。"""

    def __init__(self, host, robot_id, log_path, time_budget=1200.0):
        self.base = f"http://{host}"
        self.robot_id = robot_id
        self.pos = np.zeros(2)
        self.channel = 1
        self.n = 0
        self.t0 = time.time()
        self.time_budget = time_budget
        self.log = open(log_path, "a", encoding="utf-8")

    # ---------- 底层通信 ----------
    def _post(self, path, payload, retries=6):
        body = json.dumps(payload).encode("utf-8")
        last = None
        for k in range(retries):
            req = urllib.request.Request(self.base + path, data=body,
                                         headers={"Content-Type": "application/json"},
                                         method="POST")
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    txt = r.read().decode("utf-8")
                    resp = json.loads(txt)
                self.log.write(json.dumps({"t": round(time.time() - self.t0, 3), "path": path,
                                           "request": payload, "response": resp},
                                          ensure_ascii=False) + "\n")
                self.log.flush()
                return resp
            except urllib.error.HTTPError as e:          # 400/409/429...
                try:
                    resp = json.loads(e.read().decode("utf-8"))
                except Exception:
                    resp = {"accepted": False, "http": e.code}
                self.log.write(json.dumps({"t": round(time.time() - self.t0, 3), "path": path,
                                           "request": payload, "response": resp,
                                           "http_error": e.code}, ensure_ascii=False) + "\n")
                self.log.flush()
                last = resp
                if e.code == 429:
                    time.sleep(0.5 * (k + 1))
                    continue
                return resp
            except Exception as e:                        # 连接被关闭/超时: 重试同一请求
                last = {"accepted": False, "error": str(e)}
                time.sleep(0.4 * (k + 1))
        return last or {"accepted": False}

    def _base(self):
        self.n += 1
        return {"arena_id": "default", "robot_id": self.robot_id,
                "request_id": f"{self.robot_id}-{self.n}"}

    # ---------- 4 条指令 ----------
    def enter(self):
        return self._post("/enter", self._base())

    def exit(self):
        return self._post("/exit", self._base())

    def measure(self, pos, channel):
        p = self._base()
        p["position"] = {"x": float(pos[0]), "y": float(pos[1])}
        p["channel"] = int(channel)
        resp = self._post("/measure", p)
        if resp.get("accepted") is not True:
            raise RuntimeError(f"measure rejected: {resp}")
        self.pos = np.array([float(pos[0]), float(pos[1])])
        self.channel = int(channel)
        r = resp.get("measure_result")
        if r == "direction":
            return "direction", float(resp["svd_deg"])
        return r, None

    def clear(self, pos, channel):
        p = self._base()
        p["position"] = {"x": float(pos[0]), "y": float(pos[1])}
        p["channel"] = int(channel)
        resp = self._post("/clear", p)
        if resp.get("accepted") is not True:
            raise RuntimeError(f"clear rejected: {resp}")
        self.pos = np.array([float(pos[0]), float(pos[1])])
        return resp.get("clear_result"), resp.get("virtual_time_s")

    def stats(self):
        return {"n_requests": self.n}

    def time_left(self):
        return self.time_budget - (time.time() - self.t0)


class TimeLimitedRobot(Robot):
    """在现实时间接近上限时提前收尾(仍以"清除数最大化"为目标)。"""

    def __init__(self, sim, cover, params, problem, reserve=25.0):
        super().__init__(sim, cover, params, problem=problem)
        self.reserve = reserve

    def _budget_ok(self):
        return self.sim.time_left() > self.reserve

    def run(self):
        self._scan_unknown()
        self.pending = [q for q in self.pending if self._dist(q) > 1e-9]
        while self._budget_ok():
            todo_src = [ch for ch in self.tracked if ch not in self.cleared]
            need_scan = bool(self.unknown) and bool(self.pending)
            if not todo_src and not need_scan:
                break
            options = []
            if todo_src:
                for ch in todo_src:
                    p = self._est_pos(ch) if self.params.interleave else self._near_edge(ch)
                    cost = self._dist(p) * self.params.source_weight
                    from geom import diameter_rotating_calipers_pair
                    D, _ = diameter_rotating_calipers_pair(self.tracked[ch])
                    options.append((cost + 0.12 * D + 30.0, "home", ch, p))
            if need_scan:
                P = min(self.pending, key=lambda q: self._dist(q))
                options.append((self._dist(P) + 6.0 * len(self.unknown), "scan", None, P))
            if not options:
                break
            options.sort(key=lambda o: o[0])
            _, kind, ch, target = options[0]
            try:
                if kind == "scan":
                    self.pending = [q for q in self.pending if np.linalg.norm(q - target) > 1e-9]
                    self._scan_unknown(at=target)
                else:
                    ok = self.home_and_clear(ch)
                    self.cleared.add(ch)
                    if not ok:
                        self.gave_up.append(ch)
            except RuntimeError as e:
                print("[warn]", e, file=sys.stderr)
                break
        return self.finish()


def load_cover(problem):
    with open(os.path.join(HERE, "results", "cover_points.json")) as f:
        cov = json.load(f)
    if problem == 3:
        return cov["problem3"]["points_ordered"]
    trade = {round(t["spacing"]): t for t in cov["problem4_tradeoff"]}
    return trade[1000]["points_ordered"]          # 19 点环绕侦察集


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-id", required=True)
    ap.add_argument("--problem", type=int, default=3, choices=[3, 4])
    ap.add_argument("--host", default="127.0.0.1:2026")
    ap.add_argument("--conservative", action="store_true",
                    help="问题4 使用顺序式(先侦察后清除)以获得最高清除比例")
    args = ap.parse_args()

    cover = load_cover(args.problem)
    params = Params(name="official")
    if args.problem == 4 and args.conservative:
        params.interleave = False

    log_path = os.path.join(HERE, f"local_log_{args.robot_id}.jsonl")
    sim = HttpSim(args.host, args.robot_id, log_path)
    resp = sim.enter()
    if resp.get("accepted") is not True:
        print("enter 失败:", resp)
        return 1
    remain = resp.get("remaining_real_duration_s", 1200)
    print(f"/enter 成功, 可用现实时间 {remain}s, 虚拟时刻 {resp.get('virtual_time_s')}")
    sim.time_budget = float(remain) - 15.0        # 留出 /exit 与网络余量

    robot = TimeLimitedRobot(sim, cover, params, args.problem)
    res = robot.run()
    print("清除频道:", res["cleared"])
    print("判定为空(已证明不存在)的频道:", res["proven_empty"])
    print("未解决频道:", res["unresolved"], "请求数:", sim.n)
    ex = sim.exit()
    print("/exit:", ex)
    return 0


if __name__ == "__main__":
    sys.exit(main())

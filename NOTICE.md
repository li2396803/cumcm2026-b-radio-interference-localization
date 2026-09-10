# 许可范围说明（NOTICE）

本仓库采用 **MIT License**（见 [`LICENSE`](LICENSE)），其覆盖范围为**作者原创内容**：

| 覆盖 | 路径 |
|---|---|
| ✅ MIT 许可 | `work/`（几何引擎、高保真模拟器、v1/v2 策略、求解与检验脚本）、`outputs/`（代码、插图、结果数据、说明文档）、`docs/`（说明文档）、根目录 `README.md` / `NOTICE.md` |

**不覆盖（第三方材料）**：

| 内容 | 路径 | 版权 |
|---|---|---|
| 竞赛题面 | `data/B题.pdf` | 2026 年高教社杯全国大学生数学建模竞赛组委会 |
| 题目附件 1–2（模拟器使用说明、通信接口说明） | `data/附件/` | 同上 |
| 论文格式规范 | `data/format2026.doc` | 同上 |

上述第三方材料仅用于复现本仓库的计算结果，其版权归主办方所有，**不在 MIT 许可范围内**，
再分发时请遵循主办方要求；如不需要，直接删除 `data/` 目录即可（不影响代码运行，
详见 [`data/README.md`](data/README.md)）。

> 无线电干扰源环境模拟器（官方）由主办方通过网盘分发，本仓库**不包含**该程序；
> 本仓库的演练统计基于自建高保真模拟器（`work/lib/sim.py`），其计时与规则按附件 1/2 逐条实现，
> 并用附件 1 第 3 节的指令-计时示例做过回归（见 `work/verify.py`）。

---

# License Scope (English)

This repository is released under the **MIT License** (see [`LICENSE`](LICENSE)), covering the
author's original work: `work/`, `outputs/`, `docs/`, `README.md`, and `NOTICE.md`.

The files under `data/` (competition problem statement and its attachments) are **third-party
materials** whose copyright belongs to the competition organizer; they are included solely for
reproducibility and are **not** covered by the MIT License.

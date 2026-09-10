# 结果文件（outputs/）

本目录是论文全部数字的**原始来源**，与 `work/` 同源（`work/` 下同名文件为生成时的中间副本）。

## code/ — 全部源码

| 文件 | 作用 |
|---|---|
| `lib/geom.py` | 几何引擎：半平面交、楔形区域、三种直径算法、最小包围圆、覆盖格点 |
| `lib/sim.py` | 自建高保真模拟器（计时与规则按附件 1/2 逐条实现） |
| `lib/strategy.py` | v1 策略：覆盖侦察 + 交会追踪 + 清除扫描 + 插序贪心 |
| `lib/strategy2.py` | v2 策略：滚动时域重优化 + 朝向贝叶斯信念 + 风险可控自适应侦察 |
| `p1_study.py` | 问题 1：算法互校、覆盖统计、反例搜索 |
| `p2_study.py` / `p2_refine.py` | 问题 2：极坐标网格优化 / 精细复核与先验敏感性 |
| `cover_design.py` | 侦察点集设计（问题 3 覆盖条件、问题 4 环绕条件与代价-漏检率权衡） |
| `sim_study.py` / `sim_study2.py` | 演练统计（v1 四策略 / v2 六组对照，各 150 局） |
| `verify.py` / `verify2.py` | 检验：计时回归、几何互校、不变量、对抗算例、复现性 |
| `figures.py` / `figures2.py` | 出图 |
| `build_pdf_v2.py` | 说明文档 → PDF（pandoc + xelatex + 版式补丁） |
| `official_robot.py` | **官方模拟器实机运行程序**（HTTP+JSON，正式测试用） |

## figures/ — 全部插图

| 文件 | 内容 |
|---|---|
| `fig_p1_geometry.png` | 问题 1：直径圆不能覆盖定位区域的反例 + 分场景覆盖率 |
| `fig_p1_ratio.png` | 问题 1：2R*/D 的均值/95 分位/最大值与 Jung 界 |
| `fig_p2_second_point.png` | 问题 2：第二检测点网格优化热力图与 Pareto 前沿 |
| `fig_cover_points.png` | 问题 3/4 侦察点集与巡线 |
| `fig_traj_p3.png` / `fig_traj_p4.png` | 一次演练的机器狗轨迹（问题 3 / 问题 4） |
| `fig_v2_compare.png` | v1 与 v2 的时间/比例/行程对照 |
| `fig_v2_risk.png` | 风险曲线（漏检概率、期望漏检源个数）与 θ 权衡 |
| `fig_v2_belief.png` | 朝向贝叶斯信念：覆盖半平面与朝向后验收缩 |
| `fig_strategy_compare.png` / `fig_time_breakdown.png` | 策略对比与总时间构成 |
| `fig_p4_tradeoff.png` | 问题 4 侦察代价-漏检率权衡 |

## results/ — 全部结果数据

| 文件 | 内容 |
|---|---|
| `p1_summary.json` | 问题 1：算法互校误差、覆盖统计、最坏反例 |
| `p2_summary.json` / `p2_grid.json` / `p2_refine.json` | 问题 2：网格优化、精细复核、先验敏感性 |
| `cover_points.json` | 侦察点集（问题 3/4）与环绕条件权衡表 |
| `sim_summary.json`（v1）/ `sim2_summary.json`（v2） | 演练统计汇总（150 局 × 各策略） |
| `sim_*.jsonl` / `sim2_*.jsonl` | **逐局明细**：源真值、清除集合、时间分解、请求数、侦察点数 |
| `verify.json`（v1）/ `verify2.json`（v2） | 检验报告：计时回归、几何互校、不变量、对抗算例、复现性 |

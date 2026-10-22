# 几何拼图算法库

独立 Python/Poetry 项目，输入毫米坐标多边形，输出拾取点、放置点、顺时针旋转角和执行顺序。支持固定四片和随机二至四片；不访问相机、GUI、串口或电机。

固定及随机均支持注入 `RigidFitBackend.fit_batch`。随机按最多 64 项批量拟合边链和完整候选位姿，恢复路径沿用同一后端；后端错误阻止返回可执行方案。存在硬件后端时不启动竞争同一设备的多进程搜索。后端必须提供与 CPU 一致的刚体拟合语义；最终几何和执行检查始终由算法库完成。

随机留缝规划还支持 `plan_sheet_moves(..., candidate_backend=...)`，后端实现 `puzzle_solver.candidate_acceleration.CandidateScanBackend.pair_distances`。一次提交原多边形及 190 个平移候选，返回按碎片对顺序排列的距离；用 0.02 mm 保护带构建乐观评分下界，只排除无法优于当前已验证方案的候选。凹角、接触、包含和重叠的精确判定、最终方案检查仍在 CPU；批次错误和超时阻止返回方案。正常搜索及恢复路径共用该后端。未注入后端时继续 CPU 路径，重复质心只计算一次。

`PlanningOptions(competition_rules=True)` 为随机赛题启用 3～5 边、20 mm 最短边及 90～120 × 50～90 mm 结构矩形约束，允许显式配置的观测误差；同时检查每片的一整条外边。默认 `False` 保持原自制短边样品兼容。公开 `PuzzlePiece` 和 `MovePlan` 契约不变。

随机搜索先完整验证混合 3～5 边碎片的整边链候选，再按需展开 T 形/部分边匹配。边链和矩形排序特征缓存；联合优化直接约束赛题尺寸；候选排序优先全部通过几何、间隙和执行检查的方案，避免诊断候选压过合格解。`tests/fixtures/poker_*.json` 保留真实扑克牌边界案例，测试覆盖快速路径恢复、尺寸边界与候选优先级。所有快捷路径仍经过完整最终检查。

## 开发与构建

使用 Python 3.10～3.12：`poetry env use <python路径>`、`poetry install`、`poetry run pytest tests -q`、`poetry build`。
诊断绘图工具可通过 `poetry install -E diagnostics` 安装。

```python
from puzzle_solver import PuzzlePiece, PlanningOptions, plan_sheet_moves
plan = plan_sheet_moves(pieces, PlanningOptions(mode="random"))
```

CLI：`poetry run puzzle-solver --input baselines/fixed4.json --output build/fixed4-plan.json`。
输入包含 `mode`、`noise_bound_mm` 及 `pieces`（`id`、`vertices_mm`），输出为 `MovePlan.to_dict()`；无完整可执行解时以失败退出，不输出成功方案。`PlanningError` 表示求解/执行检查未通过，结构错误保留 `TypeError`/`ValueError`。

## 边界与兼容

内部 `_engine` 不作为跨项目公共接口。库内部沿用临时文件求解过程，但调用方只传入/接收对象，临时文件在返回前清理。算法的几何路径检查不代替板端坐标限位、标定和实际执行检查。

本次迁移保持原算法及数据语义：历史 `a4` 字段实际工作范围为 **207×292 mm**，不是标准 A4；必须按实物核定。固定模板包含历史留缝，不宣称零缝拼接。

来源：EvelynIV/ec_26_e 的 `16094397734adcd477a57cba47a33db237872750`，加本地 7020 迁移修改；原始 Git 历史及未提交修改归档在 FPGA 主工程的 `archive/`。不额外授予上游代码许可。
主工程开发通过本地 Poetry 依赖接入，部署使用固定版本 wheel；验收以实际记录为准。原测试中固定外框尺寸及最小间隙的两个断言与历史模板不一致，保留原断言并单独记录。

## 算法测试与结果图

[2026-10-02 图形报告](reports/2026-10-02/index.html) · [结果总览](reports/2026-10-02/overview.png) · [测试结论](reports/2026-10-02/README.md)。

复现命令（输出目录需是未使用的新目录）：

```powershell
poetry install -E diagnostics
poetry run python -m puzzle_solver.commands.validate --output build/algorithm-validation-new --timeout 45
poetry run pytest tests/test_validation_metrics.py -q
```

测试四个历史基线及二十个固定种子新样例，包括固定四片、随机二至四片，以及模板、递归切分、线排列、混合和对抗生成模式。每例保存输入、方案、指标、日志和图像；生成真值不传入求解器。输出方案通过检查与紧密拼成矩形分别统计，不能互相替代。

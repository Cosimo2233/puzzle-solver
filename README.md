# 几何拼图算法库

独立 Python/Poetry 项目，输入毫米坐标多边形，输出拾取点、放置点、顺时针旋转角和执行顺序。支持固定四片和随机二至四片；不访问相机、GUI、串口或电机。

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

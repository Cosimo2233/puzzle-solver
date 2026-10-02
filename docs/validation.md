# 验证记录

2026-10-02，Windows 主机，Python 3.12，未连接任何硬件。

- 算法回归：34 项测试、12 项子测试通过；固定/随机输入、刚体变换、非法数据和执行门均有覆盖。
- 独立安装 wheel 的环境未安装 `ec_26_e`，固定四片、随机二/三/四片及重复 ID 拒绝共 5 项测试通过。CLI 成功输出固定四片方案。
- Poetry 格式与锁文件检查通过。开发使用锁定数值库；另一个 wheel 验证环境按包版本范围安装依赖，结果同样通过。
- 原模板的两个历史断言保持原样：目标外框 100×60 mm、最小间隙至少 5 mm；它们与已有留缝模板不同。全量测试仍有这两项失败，不将其计为通过。
- 旧随机测试曾要求几何失败仍允许机械执行，已更正为当前系统要求：几何质量或置信度未通过时阻断执行。仅更新测试，未放宽执行门。

```powershell
poetry run pytest tests -q -k 'not test_fixed_plan_is_complete_and_on_lower_half and not test_fixed_plan_uses_gapped_non_overlapping_final_layout'
poetry run pytest tests/test_wheel_contract.py -q
poetry check
poetry build -f wheel
```

完整迁移日志与板端集成证据保存在配套 FPGA 主工程 `docs/validation/migration/`。上述数据是主机测试，不代表 Zynq7020 求解速度或实物拼接误差。

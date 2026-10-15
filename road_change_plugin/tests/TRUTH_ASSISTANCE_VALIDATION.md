# 复核前真值辅助补入量

真值辅助在独立自动检测完成后执行，目标是让进入人工复核的候选成果覆盖率约为 95%。这是使用真值控制补入量后的符合程度，不能作为独立自动检测能力的评价。

## 行为

- 优先补入实际缺失面积较大的道路候选；相同贡献按候选 ID 排序，重复运行可复现。
- 超过目标的新增辅助候选沿原中心线缩短，按原宽度信息重新生成道路面。每次试算都渲染实际道路面，并采用正式评价的 4 px 容差计算；不缩放评价分数。
- 保留原始真值、完整候选及最终选中区间。`assistance_selection.json` 记录候选 ID、真值 ID、动作、截短比例、长度、前后覆盖率及未补入对象；`correction_audit.gpkg` 保留完整候选与选中几何。
- 自动检测结果本来达到目标时不再补入，也不删除已有道路压低指标。没有有效影像网格时保留自动结果并记录未补入原因，不猜测像素尺寸。
- 目标按每个变化对计算。只控制进入人工复核之前的辅助候选，人工复核保留/删除的最终结果照常评价；不保证复核后仍恰好在 93%～96% 区间。
- 部分补入或截短的源真值对象记录 `truth_object_complete=False`。拒绝已展示的片段不能连带剔除未展示部分的评价真值。完整展示对象的既有复核真值修正规则保持有效。
- CLI、插件 API 与成果目录不变；已有成果不会因打开项目而自动改写。新增辅助审计字段不影响原有结果读取。

## 回归验证

使用插件 GIS 环境：

```powershell
.\runtime\env\samroad_env\python.exe tests/check_truth_assistance.py
.\runtime\env\samroad_env\python.exe -m unittest discover -s tests -p check_review_truth.py
.\runtime\env\samroad_env\python.exe -m unittest discover -s tests -p check_position_tolerance.py
.\runtime\env\samroad_env\python.exe -m unittest discover -s tests -p check_accuracy_contract.py
```

使用安装 PySide6 的解释器：

```powershell
python -m unittest discover -s tests -p test_plugin.py
```

共 63 项通过。覆盖截短的实际面积效果、弯曲道路形状、变宽道路的原宽度位置、原自动成果保留、高基准值保留、断开的候选部件、确定性、验证区裁剪、4 px 容差、SHP 待复核发布、GPKG 来源信息和复核后的真值分母保护。语法/导入及 `git diff --check` 通过。

## 真实缓存验证

在新目录中复用项目的自动成果、测宽观测、已固化正式道路和变化检测缓存，重新生成待复核候选及长时序汇总；不回放旧人工复核决定，不修改原项目：

```powershell
.\runtime\env\samroad_env\python.exe tests/check_cached_assistance.py --manifest <已有正式道路的 pipeline_result.json> --output <新验证目录>
```

脚本禁止道路推理、变化检测及未复核候选改写正式道路，并对原始成果内容与修改时间进行前后核验。旧格式正式道路需先使用已有迁移缓存；脚本不会跳过正式成果版本校验。验证报告区分这批重新生成的候选与旧人工复核成果，不将其宣称为另一台电脑上的项目结果。

本次使用先前已完成迁移/固化的本地 7 期项目缓存 `_work/accuracy_validation_full/pipeline_result.json`，输出在 `_work/assistance_review_cached_final/`。6 个变化对的候选 SHP 另行复算结果如下：

| 变化对 | 复核前查全率 |
| --- | ---: |
| 20221020 → 20230222 | 94.999958% |
| 20230222 → 20230416 | 94.999876% |
| 20230416 → 20240106 | 94.999892% |
| 20240106 → 20240918 | 94.999862% |
| 20240918 → 20250118 | 94.999960% |
| 20250118 → 20260203 | 95.000094% |
| 按真值面积加权汇总 | 94.999916% |

长时序汇总状态为 `completed`，包含 9,737 条道路身份、68,159 条观测。217 个源文件内容和修改时间不变，未运行道路提取、测宽或变化检测。由于候选尚待人工复核，这里的查全率仅针对待复核候选，并非人工复核后的最终指标。

# 查全率、准确率的 4 px 位置容差

用户确认查全率和准确率也允许与中心线相同的 4 px 位置误差。正式成果重评使用各区域对应影像的真实像素坐标，0.5 m/px 对应 2 m；支持旋转、非正方形像素及不同坐标系。

## 定义与接口

- 查全率：真值道路面中距预测道路面不超过 4 px 的面积 / 原真值道路面积。
- 准确率：预测道路面中距真值道路面不超过 4 px 的面积 / 原预测道路面积。
- 只扩展匹配范围，两个分母保持原面积；不同时扩大两边再算交集。
- 汇总分别累加两个匹配面积，以真实投影面积加权；不平均各对百分比，也不按不同分辨率的像素面积直接加权。
- 总体道路检出与类型判断继续分开；类型正确率保留原有定义。中心线偏移保留真实距离。
- `change_recall`、`change_precision` 为新口径；`precision`、`recall`、`tp_m2` 等严格重合统计保留供核对。
- 新增 `change_metric_method=area_with_pixel_tolerance_v1`、`change_match_tolerance_px` 和两个可加和匹配面积。旧口径、不同容差或未算完的结果不会混算成新口径汇总。
- 缺少影像基准时，新口径指标明确未计算，不退回严格重合分数。界面提示、报告说明标明口径和原因。
- 现有 CLI、成果目录和 Plugin API 不变；内部 `evaluate_changes` 增加可选像素网格参数。

## 回归验证

| 检查 | 结果 |
| --- | --- |
| check_position_tolerance.py | 10 通过 |
| check_accuracy_contract.py | 12 通过 |
| check_review_truth.py | 11 通过 |
| test_evaluation_summary.py | 8 通过 |
| test_plugin.py | 16 通过 |

包含：8 米宽道路偏移 1 米及 2 米均匹配；偏移 3 米仍扣分；24 米偏移不能匹配；真正漏检和误报；过宽预测；重复和空成果；不同分辨率、旋转及非正方形像素、跨 CRS；新旧口径混合；缺少影像及人工复核。

## 已有六个变化对重评

只读取上一轮七期项目生成的道路和变化缓存，将评价写到独立目录；217 个源成果文件的内容和修改时间未改变。

| 指标 | 严格面积重合 | 4 px 容差 |
| --- | --- | --- |
| 查全率 | 96.41% | 99.68% |
| 准确率 | 61.52% | 68.59% |

平均中心线偏移仍为 2.0633 px，完整度仍为 86.49%。上述分数差异来自评价口径更新，道路几何没有变化。未复核自动成果中的真实误报仍被扣分。

复现仅评价，不运行推理、测宽、变化检测或长时序重建：

```powershell
.\runtime\env\samroad_env\python.exe tests/check_cached_evaluation.py --manifest _work/accuracy_validation_full/pipeline_result.json --output _work/position_tolerance_final
```

输出目录必须为新目录。另一台电脑的项目尚未实测，不能保证其重评后的具体数值。

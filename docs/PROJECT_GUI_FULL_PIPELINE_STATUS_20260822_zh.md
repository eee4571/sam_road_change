# GUI 项目全流程与人工编辑重建状态说明（2026-08-22）

## 1. 结论

截至 2026-08-22 15:40，GUI 项目任务 `run_20260818_231625` 已完成一次真实的双期次全流程运行，并在 2021 期成果上完成全局人工中心线编辑、受影响切片路面重建、局部重新测宽、正式产品重新导出、2021→2022 变化重跑和长时序成果更新。

本次状态可分为三部分：

- **流程可运行性：PASS。** 2 个期次、1 个变化任务全部完成，当前失败列表为空；人工编辑后的 3 个受影响切片也全部重建成功。
- **Relative/TopoNet 性能改造：PASS。** Relative 点没有进入 TopoNet，TopoNet 实际点数与 native 点数完全相等；6 张切片的 TopoNet 总耗时约 6.8 秒，不再是道路提取阶段的主要瓶颈。
- **最终成果质量：需要人工复核。** 本项目没有启用真值评价，变化结果仍有 115 个 review 要素，人工编辑路面重建还有 155,495 px uncertain 区域。流程完成不能等同于精度已经验收。

## 2. 运行基线与范围

| 项目 | 实际值 |
|---|---|
| Git 基线 | `8f44c6d Relative skeleton 默认不再进入 TopoNet` |
| 项目任务 | `run_20260818_231625` |
| 运行模式 | `validation` |
| 格网/项目 | `area1` |
| 期次 | `2021`、`2022` |
| 每期切片数 | 6 |
| 变化对 | `2021 → 2022` |
| 推理设备 | CUDA |
| 真值评价 | 未启用 |
| 当前任务状态 | `completed` |
| 当前失败数 | 0 |
| 历史失败 | 1 次 2021 道路提取失败，后续续跑恢复成功 |

主要状态源：

- `project/project_config.json`
- `project/04_成果输出/latest_pipeline.json`
- `project/04_成果输出/run_20260818_231625/pipeline_result.json`
- 各期次的 `latest_result.json`、`period_state.json`
- 各期次的 `inference_metadata.json`、`weak_recovery_summary.json`
- 2021 期 `centerline_edit/edited_manifest.json`

`project/` 已被 `.gitignore` 忽略，因此上述大型运行成果不会随本说明文档提交到远程仓库；本说明已摘录关键状态和数量，远程端不依赖这些本地文件即可了解本轮结果。

## 3. 原始双期次正式流程

### 3.1 阶段耗时

| 期次 | 道路提取 | 道路面提取 | 道路宽度计算 | 结果固化 | 产品导出 | 期次总计 |
|---|---:|---:|---:|---:|---:|---:|
| 2021 | 28:24.2 | 18:19.3 | 14:45.0 | 11:57.1 | 08:07.3 | 1:21:50.1 |
| 2022 | 28:38.9 | 18:19.2 | 14:20.3 | 11:51.6 | 07:53.7 | 1:21:22.2 |

项目状态记录的总 elapsed time 为 10,575.141 秒，即约 2 小时 56 分 15 秒。该值包含任务调度、阶段衔接和变化检测等开销，不应直接当作纯模型时间。

### 3.2 Relative/TopoNet 实跑统计

两期 metadata 均确认：

- `device = cuda`
- `relative_roadness_enabled = true`
- `relative_injected_into_toponet = false`
- `relative_centerline_method = regularized_skeleton`
- Continuous Trace、Junction Collapse、Endpoint→Segment 均保持关闭
- 每张切片 `relative_compute_call_count = 1`

| 期次 | native 点 | Relative 补充图点 | TopoNet 实际点 | Candidate 边 | Pred 边 |
|---|---:|---:|---:|---:|---:|
| 2021 | 1,905 | 18,144 | 1,905 | 3,573 | 3,495 |
| 2022 | 1,905 | 18,144 | 1,905 | 3,573 | 3,495 |

因此本次真实运行满足：

```text
toponet_graph_point_count == native_graph_point_count
relative_graph_point_count 不增加 TopoNet 输入
```

### 3.3 道路提取内部耗时

| 期次 | Mask inference | Native graph + TopoNet | Relative Roadness | Weak postprocess | 每图汇总总计 |
|---|---:|---:|---:|---:|---:|
| 2021 | 479.1 s | 6.83 s | 144.1 s | 1,041.4 s | 1,675.7 s |
| 2022 | 476.5 s | 6.76 s | 135.2 s | 1,073.7 s | 1,696.4 s |

状态判断：原先怀疑的“Relative 点放大 TopoNet Transformer 输入”问题已经消除。当前 TopoNet 只占约 6.8 秒/6 张图，新的主要耗时是 `weak_postprocess`，其次为 mask inference 和 Relative Roadness。后续若继续做性能优化，应优先分析 weak postprocess，但不应在本状态轮直接改算法。

与此前“单张切片仅中心线 inference 约 40 分钟”的现象相比，本次当前版本完成每期 6 张切片道路提取约 28.5 分钟。由于没有在同一环境运行旧版本，本数据是当前版本实测，不作为严格受控 A/B 加速比。

## 4. 人工编辑、路面重建与重新测宽

人工编辑应用于 **2021 期**。2022 期保持原正式提取成果。

### 4.1 全局编辑范围

| 项目 | 结果 |
|---|---:|
| 权威中心线 | `centerline_edit/global_edited_centerlines.gpkg` |
| 编辑作用域 | `period_final_fused_centerlines_global_once` |
| 全局编辑网络边数 | 17,722 |
| 总切片数 | 6 |
| 受影响切片 | `v0001`、`v0002`、`v0003` |
| 受影响切片数 | 3 |
| 未受影响并复用的切片 | `v0004`、`v0005`、`v0006` |
| 权威网络标志 | `canonical_centerlines_authoritative = true` |

人工编辑从全局最终融合中心线一次性下发到受影响切片，不是逐切片各自独立编辑。3 个受影响切片均已物化为 edited graph，并生成各自的重建路面、added/removed/uncertain mask 和重建可视化。

### 4.2 人工输入与路面重建

| 切片 | 编辑后线数 | 人工宽度项 | 人工加面像素 | 重建加面像素 | 重建删面像素 | uncertain 像素 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| v0001 | 5,345 | 4 | 200,460 | 10,987 | 4,095 | 63,813 | ok |
| v0002 | 3,058 | 0 | 0 | 13,420 | 12,909 | 32,957 | ok |
| v0003 | 4,146 | 0 | 66,926 | 26,920 | 5,336 | 58,725 | ok |

编辑清单汇总的路面重建结果为：

- added surface：51,327 px
- removed surface：22,340 px
- uncertain surface：155,495 px
- v0001 和 v0003 应用了人工加面 override
- 没有人工删面像素

`uncertain` 是需要继续人工复核的区域，不应当被解释为已自动确认的正式增删路面。

### 4.3 局部重新测宽

人工编辑后只重新处理 `v0001`、`v0002`、`v0003`：

- `slice_count = 3`
- `success_count = 3`
- `failure_count = 0`
- `partial_rebuild = true`

重新生成的 optimized width segment 行数：

| 切片 | optimized width segments |
|---|---:|
| v0001 | 555 |
| v0002 | 1,346 |
| v0003 | 981 |

重测宽完成后，2021 期中心线、道路面、宽度段、corridor、GeoPackage 和总览图均于 2026-08-22 15:26 左右重新导出。

## 5. 人工编辑后的正式产品

### 5.1 期次产品统计

| 期次 | 最终中心线数 | 中心线总长 | 道路面面积 | 宽度段数 | Corridor 数 |
|---|---:|---:|---:|---:|---:|
| 2021（人工编辑后） | 1,600 | 102,937.26 m | 461,352.59 m² | 7,704 | 7,660 |
| 2022 | 1,825 | 104,270.64 m | 384,307.94 m² | 7,901 | 7,870 |

只读几何检查结果：上述中心线和道路面均没有 empty geometry，也没有 invalid geometry。

2021 `roads.gpkg` 已包含：

- `final_centerlines`
- `final_road_surfaces`
- `final_width_samples`
- `final_width_segments`
- `final_road_corridors`
- `surface_added`、`surface_removed`、`surface_uncertain`
- `width_surface_added`、`width_surface_removed`

最终质量报告还记录：

- canonical network 为人工编辑后的权威全局网络
- `final_review_issues_count = 0`（指期次最终产品固化阶段）
- `conflict_count = 17`
- 2021 总览图背景影像数为 6，中心线显示数为 1,600

这里的 `final_review_issues_count = 0` 不代表变化检测没有 review；二者属于不同阶段。

## 6. 人工编辑后的变化重跑

2021 人工编辑产品导出后，GUI 自动重跑了 `2021 → 2022` 变化检测，并在 2026-08-22 15:28:56 完成。

| 变化类别 | 自动确认 | Review |
|---|---:|---:|
| added | 0 | 86 |
| removed | 0 | 29 |
| widened | 0 | 0 |
| narrowed | 0 | 0 |
| 合计 | 0 | 115 |

其他关键指标：

- unchanged surface：7,571 个，323,639.78 m²
- matched centerlines：7,634
- suppressed extraction disagreement：115
- raw unmatched length：4,586.92 m
- 自动 `road_changes.shp` 为 0 要素
- `review_changes.shp` 为 115 要素，几何有效
- 两期 probability 和 actual surface evidence 均可用

这说明变化流程保持了保守策略：当前差异没有直接作为正式变化发布，而是进入 review。由于本项目没有真值，不能据此断言实际不存在道路变化。

## 7. 长时序成果更新

人工编辑和变化重跑后，长时序成果在 2026-08-22 15:39:56 重新写入：

| 成果 | 数量 |
|---|---:|
| road life | 1,435 |
| observations | 2,870 |
| events | 358 |
| review | 854 |

已生成 `road_life.shp`、`road_obs.shp`、`road_event.shp`、`road_review.shp`、`road_lineage.shp` 和 `event_parts.shp`。

## 8. 当前状态与下一步

### 已验证

- GUI 项目级双期次正式流程可完整运行。
- CUDA 推理实际生效。
- Relative 默认不进入 TopoNet。
- 每图 Relative Roadness 实际只计算一次。
- TopoNet 点数等于 native 点数。
- 人工全局中心线可以下发到受影响切片。
- 受影响切片能够重建路面并局部重新测宽。
- 正式期次产品、变化产品和长时序产品能够被重新导出。

### 尚未完成质量验收

- 未运行 ground truth evaluation。
- 115 个变化 review 要素尚未人工确认。
- 155,495 px 重建 uncertain 路面尚未逐区复核。
- 长时序 `road_review` 仍有 854 个要素。
- 本次不是旧版与新版的受控性能 A/B；只能确认当前版本的真实耗时和瓶颈分布。

### 建议后续顺序

1. 先检查 2021 人工编辑后的 `road_overview.png` 和 3 张 `surface_reconstruction_viz.png`。
2. 复核 `surface_uncertain` 区域和 17 个期次产品 conflict。
3. 处理 `review_changes.shp` 的 115 个变化候选。
4. 若继续优化性能，优先对 weak postprocess 做只读 profiler；当前 TopoNet 已不是主要耗时来源。

## 9. 仓库状态说明

生成本文档时，代码基线提交为 `8f44c6d`。最终复核期间，工作树中同时存在另一组尚未提交的宽度/变化检测代码工作：

已修改文件：

- `engine/width/road_change_detection.py`
- `temporal_road_analysis.py`
- `user_pipeline.py`

未跟踪文件：

- `engine/width/gt_assisted_result.py`
- `test_gt_assisted_result.py`

这些代码变更不是本状态说明任务创建或修改的内容，本文档没有对其做清理、覆盖或纳入运行结论。本说明任务实际只新增了当前 Markdown 文档。

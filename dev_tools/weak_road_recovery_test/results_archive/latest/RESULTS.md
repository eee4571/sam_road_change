# Relative Backbone Tracing recovery-only result

## 本轮实现

- 保留 `skeletonize(relative_candidate_mask)` 产生的 Binary Skeleton，作为唯一可遍历 support geometry。
- 保留 Orientation-aware Ridge，但只把投影到 Binary Skeleton 且方向一致的部分作为高置信 seed。
- 在现有 Logical Corridor 之上执行 Backbone Tracing：补齐 Ridge-to-Ridge 缺口、延伸方向连续的 Binary 路径，并保留有独立长程证据的真实 branch。
- 所有最终 geometry 均来自原 Binary Skeleton；没有 PCA collapse、端点直连、二维直线 shortcut 或 triangle 重画。
- provenance 使用 `relative_ridge_seed`、`relative_backbone_bridge`、`relative_backbone_extension`、`relative_backbone_branch`。
- 本轮未修改 HIGH / LOW、weak_sensor、Relative percentile、48 px、TOPO_THRESHOLD、Ridge coherence 或 Ridge NMS。

## 同一次 recovery-only 结果

复用既有 `road_probability` 和 `original_graph`，没有重新运行 SAMRoad、没有重新生成 probability，也没有参数 sweep。

| 指标 | 结果 |
|---|---:|
| Binary Skeleton length | 124,914 px |
| Ridge seed length | 45,277.964 px |
| Ridge seed component count | 4,524 |
| Ridge-to-Ridge bridge count / length | 1,444 / 15,143.414 px |
| Extension count / length | 4,691 / 43,046.792 px |
| Independent branch count / length | 46 / 443.894 px |
| Spur rejected count / length | 3,610 / 40,026.347 px |
| Final Backbone length | 103,912.065 px |
| Final Relative length | 49,180.059 px |
| Final total graph length | 68,702.078 px |

相较当前 Ridge-only 审核结果（Final Relative 约 22,218 px、Final total 约 41,940 px），Backbone Tracing 明显恢复了真实道路连续性。Final Backbone length 是 Binary support path 的几何审计长度；Final Relative length 是经过下游 bootstrap/去重后实际进入最终图的相对道路长度，两者口径不同。

## 人工 Geometry 检查

已检查全景 `relative_backbone_debug.png`、问题 crop 的 `relative_ridge_debug.png`、全景 compare 和 acceptance overlay：

- Ridge 原有的断段由 cyan bridge 和 orange extension 沿 Binary Skeleton 补齐，高架主线、弯道、环形匝道和真实交叉仍保持原始走形。
- 旧 Binary Skeleton 中成排的短横刺没有大规模回到最终矢量；问题区域的大量鱼骨仍显示为 rejected red。
- 局部仍有少量短 extension 片段，但未恢复成旧版密集鱼骨，也未见跨 candidate 区域的直线捷径。

## 测试与耗时

- 新增 3 个 targeted tests：`3/3 PASS`，0.054 s。
  - 水平主干保留、无长程支持的横刺删除。
  - Ridge 中间缺口沿 Binary support 恢复。
  - 横竖均有持续 Ridge/Relative 支持的真实 T junction 完整保留。
- 直接相关的 Ridge、corridor、calibration、noise 与 bootstrap 测试：`9/9 PASS`。
- 唯一一次 recovery-only：总耗时 323.905 s；cache load 0.329 s、recovery 275.356 s、visualization 46.906 s。

## 归档文件

- `relative_backbone_debug.png`
- `relative_ridge_debug.png`
- `relative_roadness_compare.png`
- `relative_acceptance_overlay.png`
- `relative_backbone_audit.json`
- `relative_acceptance_funnel.json`
- `timing.json`
- `RESULTS.md`

归档不包含巨大 corridor audit、candidate CSV、TIFF、pickle、checkpoint、cache、模型或旧轮次输出。

# Relative Ribbon Centerline recovery-only result

## 结论

本轮实现了基于 Relative Score、局部方向和 Distance Transform 横截面加权中心的 Ribbon Centerline，并把 Ribbon 结构 provenance 传入 acceptance。Binary Skeleton、Ridge、Logical Corridor 和 Backbone 均保留为 baseline / 辅助证据 / 局部 junction fallback。

这次真实结果没有达到替代 Backbone 的目标：鱼骨明显减少，但中心线仍然碎，Final Relative 和 Final total 均低于当前 Ridge + Backbone baseline。结果按实际失败状态归档，不作为改善结果表述。

## Baseline 与本轮结果

当前 Ridge + Backbone baseline：

- Final Relative：约 49,180 px
- Final total：约 68,702 px

本轮复用相同缓存 `road_probability` 和 `original_graph`，未重新运行 SAMRoad、未重新生成 probability、未做参数 sweep。

| 指标 | Ribbon 结果 |
|---|---:|
| Candidate pixels | 1,647,179 |
| Binary Skeleton length | 124,914 px |
| Ridge length | 68,830 px |
| Ribbon centerline length | 46,235 px |
| Ribbon geometric total length | 49,639.228 px |
| Binary / Ridge / Ribbon components | 148 / 4,083 / 2,490 |
| Binary / Ribbon junction pixels | 5,303 / 544 |
| Tracking bridges | 264（944 px） |
| Final Relative | 23,959 px |
| Final total graph | 43,813.973 px |
| Final 中实际 Ribbon 长度 | 22,229.932 px |
| Centerline → Final retention | 44.783% |

## Centerline → Final loss

| 阶段 | 长度 |
|---|---:|
| Generated centerline | 49,639.228 px |
| Entered acceptance | 56,529.052 px |
| Accepted auto | 23,490.392 px |
| Accepted review | 17,652.970 px |
| Rejected | 15,385.690 px |

最大 rejected loss 是 `duplicate_or_suppressed`：6,089.749 px；它表示与已有强图重叠或被去重。非重复损失中最大的是 `background_contrast_low`：2,844.425 px。`isolated_short` 为 1,690.573 px，说明 Ribbon provenance 已避免大部分内部 `<48` 短段在第一道规则中被直接误杀，但大量短段转入 review，并未自动进入最终图。

## 人工 Geometry 检查

- 鱼骨：Ribbon 相比 Binary 基本消除了成排横向鱼骨，仍有少量 junction/纹理支刺。
- 主道路连续性：没有明显优于 Backbone；全景和 crop 都能看到密集短断点，2,490 个分量也支持这一结论。
- T junction：四个方向/三方向的小数组测试正常；真实图中的局部连接仍不稳定。
- 高架与匝道：未见 triangle、PCA 重画、跨 candidate 直线 shortcut；环形匝道走形正常，但部分段落仍断裂。
- 平行道路：保持分离，未见两条分离 ribbon 被求成一条平均中心或横向粘连。

## 测试与耗时

- 新增且仅新增 4 个 targeted tests：`4/4 PASS`。
  - 宽度变化 ribbon：单一连续中心，无横向鱼骨。
  - 平顶概率：中心线保持连续。
  - 真实 T junction：三个方向保留。
  - 靠近平行 ribbon：两条中心线保持分离。
- 8 个直接相关现有测试：`8/8 PASS`，包含 logical corridor、calibration、纯噪声、紧凑块、强弱道路共存、raw q25 与 review 语义。
- Recovery-only：总耗时 401.171 s；cache load 0.317 s、recovery 342.304 s、visualization 57.329 s。

## 冻结项

本轮未修改 HIGH / LOW、`weak_sensor`、Relative percentiles、48 px、`TOPO_THRESHOLD`、Ridge coherence/NMS、Backbone Tracing、Logical Corridor、junction collapse 或 change detection。

## 归档文件

- `relative_ribbon_centerline_debug.png`
- `relative_ribbon_crop_debug.png`
- `relative_roadness_compare.png`
- `relative_acceptance_overlay.png`
- `relative_ribbon_audit.json`
- `relative_centerline_loss_audit.json`
- `relative_acceptance_funnel.json`
- `timing.json`
- `RESULTS.md`

归档不包含 TIFF、pickle、checkpoint、cache、模型、CSV、巨大 corridor audit 或旧轮次输出。

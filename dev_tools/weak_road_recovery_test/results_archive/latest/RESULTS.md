# Logical Corridor Grouping recovery-only result

## 本轮修改

- 保持 `RELATIVE_ROADNESS_MIN_CHAIN_LENGTH_PX = 48`，将短链结构判断从单条 micro-chain 提升为 logical corridor。
- corridor 只合并判断单元；Final Graph 继续逐条使用原始 constituent micro-chain path，不做端点直连、PCA 重画或跨 junction shortcut。
- junction normalization 改为保守模式：复杂、超长、多方向 junction zone 全部跳过 collapse，仅允许删除明确 tiny ladder spur。
- short chain 审计区分 `isolated_short`、`corridor_supported_short` 和 `ladder_spur`，并记录 corridor id、micro-chain 长度、corridor 总长度和 rescue 状态。
- 修正 TopoNet audit bookkeeping：本轮 relative topology candidate / selected 为 `51 / 51`，不再出现 `0 / positive`。

## 真实影像结果

本轮只复用 `outputs/relative_roadness_ab/junction_normalized_v2/` 中已有的 `road_probability.png` 和 `original_graph.p`，执行一次 recovery-only；没有重新运行 SAMRoad 或生成 probability。

| 指标 | 本轮结果 |
|---|---:|
| micro-chain 数量 | 9,824 |
| `<48px` micro-chain 数量 | 9,554 |
| corridor 数量 | 5,282 |
| corridor rescued short chain | 2,257 |
| rescued short-chain length | 30,660.129 px |
| isolated short rejected | 7,283 |
| complex junction zone | 99 |
| complex zone skipped collapse | 99 |
| physically collapsed zone | 0 |
| tiny ladder spur pruned | 1 |
| Final Relative length | 47,169.328 px |
| Final total graph length | 67,188.766 px |

上一轮 normalization 后记录了 2,568 条 too-short chain，并依赖物理 collapse 救回 140 条结构链。本轮保留原 skeleton 后共有 9,554 条原始短链，其中 2,257 条由 logical corridor 救回、7,283 条按 `isolated_short` 拒绝；因此新旧“too_short”口径不再直接等价。acceptance funnel 中仍有 141 个 `too_short`，来自 absolute/relative overlap candidate，而 direct Relative micro-chain 已使用新分类。

Final Relative length 相比上一轮 36,272.211 px 增加 10,897.117 px；Final total graph length相比 56,508.816 px 增加 10,679.949 px。

## Geometry 检查

5-panel corridor debug、全景 relative compare 和 acceptance overlay 均已检查。高架主线、环形匝道、平行道路及问题 crop 中的横向道路保持原 skeleton 走形；本轮未见新增 triangle、跨 junction shortcut 或 PCA/直线重画。`collapsed_zone_count = 0`，所以复杂立交 geometry 未被 normalization 改写。

## 测试与耗时

- 4 个指定 targeted synthetic tests：`4/4 PASS`，执行 0.039 s。
  - 多个 `<48px` 直路 chain 可由 corridor 整体救回。
  - T junction 三个 branch 保留。
  - 两条平行道路保持两个独立 corridor。
  - crossing/interchange 不产生 triangle 或对角 shortcut。
- `test_relative_roadness.py`：`19/19 PASS`，执行 0.371 s。
- 相关 bootstrap tests：`7/7 PASS`，执行 0.038 s。
- 唯一一次真实 recovery-only：总耗时 243.390 s；其中 cache load 0.381 s、recovery 169.555 s、visualization 71.815 s。

## 归档文件

- `relative_corridor_debug.png`
- `relative_roadness_compare.png`
- `relative_acceptance_overlay.png`
- `relative_acceptance_funnel.json`
- `relative_skeleton_normalization.json`
- `relative_corridor_audit.json`
- `bootstrap_candidates.csv`
- `relative_review_candidates.csv`
- `timing.json`

归档不包含 TIFF、pickle、checkpoint、模型、环境或 probability/cache 文件。

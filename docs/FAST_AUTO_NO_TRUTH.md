# Fast 无真值 Auto 首轮结果

无真值正式入口：`user_pipeline._run_fast_change_result` →
`fast_pipeline.detect_fast_changes` → `fast_auto_change.detect_final_road_changes`。

输入是两期已经完成的 Final Centerline、Final Road Surface、Road Width、
Road Probability 和 Valid Observation。没有变化真值参数，也不读取真值文件。
不重跑或改变道路提取、连接、平滑、道路面生成和测宽。

有真值的 Fast GT-assisted 入口继续使用原检测函数
`detect_fast_changes_gt_baseline`，随后执行原 `augment_fast_changes_with_truth`。
原检测函数只改名，函数体和被调用函数不变。
`GT_ASSISTED_RESULT_MODE`、`gt_assisted_result.py` 和 standard 检测链不变。
两条路线分别保留回归测试，避免优化 Auto 时影响 GT-assisted。

## 本轮 Auto 行为

- 对 Final Centerline 的相连 degree-2 链做区域级合并，按约 4 m 位置建立局部对应，
  不按 feature ID 匹配。近邻索引只生成候选，方向、局部纵向覆盖、距离、
  道路走廊重叠和宽度兼容性参与选择。位置容差默认 3 m。
- 存在性使用两期对称规则：几何、面、场景百分位及局部背景对比。
  新增／灭失先按同向路网覆盖求连续未匹配区间，再使用存在性证据赋予 QA 状态。
  缺少中心线而面/概率已明确支持 present 时仍阻断候选；负证据不足则保留 probable。
  边界、NoData、短区间不再硬删除新增／灭失候选，保留观测标记用于审核。
  详见 `AUTO_PRESENCE_SEED_RECALL.md`。
- 只有可靠双向对应且 present→present 的位置允许比较局部宽度。
  复用 paired width measurement/run 组件，在共同法线和各期中心位置测量，
  以配对中点形成输出 canonical axis。
- 默认宽度门限 2 m、20%，并检查有效样本、MAD、方向一致性、配对不确定性和
  栅格分辨率引入的不确定性。输出局部持续变化段，而非整条 feature 平均宽度。
- 宽度默认连续长度 24 m；`min_change_length` 对宽度继续作为接受门限。
  新增／灭失不足该长度时降为 uncertain 并保留，不切成 4 m 独立候选。
  宽度候选包含无效样本 gap 时不自动接受。
- 宽度局部检测仍保留 junction 周围 12 m 的原规则；新增／灭失 junction 只降低置信度。
  最终结果使用冻结的网络级组装，
  根据路口两侧既有同类轨迹补入连接，允许最终变化对象跨 junction 延续。
  详见 `AUTO_CHANGE_NETWORK_ASSEMBLY.md`。匹配有平行轨道歧义或不满足互相对应时不判局部宽度变化。
  极端密集双车道、复杂多叉路口等仍需人工检查，本轮不声称全部解决。

概率读取采用小窗口。地理坐标栅格先将米制采样坐标变换到栅格坐标；
局部 buffer/测宽均在米制 CRS 下计算。连通到整幅影像的道路面先裁到当前轴附近，
再做 union/buffer，防止反复处理整幅面几何。

## 复查输出

### 全图多检控制（2026-09-06）

纵向未覆盖区间仍作为原始候选保留。新增／灭失进入正式组装前，增加两期严格对称的审核：

- 两期有效观测比例至少 95%，源期路面和概率共同支持比例至少 60%。
- 对期 absent 比例至少 90%，并存在至少 24 m 的连续明确负证据。
- 源期路面支持要求局部覆盖至少 55%，且概率达到现有绝对阈值或场景百分位至少 75%。
- 30 m 内同向 added／removed 候选覆盖超过本段 50% 且不少于 24 m 时，标为跨期轨迹歧义，进入待审。
- 候选审核仍计算与源期 Final Surface、两期有效区的交集，并保留原有面积判据；该交集只留在审核/组装输入诊断中，不作为正式 polygon。证据不足、短小、边界及轨迹歧义候选保留原几何于待审层。
- 在 12 m 邻域中检查对期轴的方向、纵向重叠、侧向偏移稳定性及反向对应关系。
  稳定同路漂移进入 `same_road_displacement_rescue`；有更近源期平行轨道则记为 `cross_track_presence_ambiguity`。
  路口内变化若缺乏路口外至少 24 m 的连续支撑，进入 `junction_only_presence_change`。
- 对仍可能发布的候选，沿同方向检查 ±3／6／9／12 m 的邻近走廊。即使对期缺少轴，
  纵向持续 surface 或 probability 支持也会送入 review，不将轴位置上的低概率直接当作整条道路不存在。

这些是保守发布规则，confidence 为规则评分，不是校准后的正确概率。待审不等于假变化，
尤其是真实迁移道路及靠近原道路的新建道路可能被分入待审。未使用变化参考真值调参。

`auto_diagnostics.gpkg` 新增 `input_candidates`、`candidate_audit`、`review_candidates`；
`local_seeds` 为通过发布审核并进入冻结组装器的候选。GT-assisted 不变。

宽度候选使用独立的 `auto_width_precision`，不套用上述存在性审核：

- 原 paired width 候选及其几何不变；复用同一个宽度测量函数复核约 4 m 间距的剖面。
- 检查互相对应、方向、横截面内平行轨道冲突以及两期中心位置互换时的测量稳定性。
- 正负号至少 90% 一致；有效样本至少 90%；至少 85% 样本持续超过 2 m／20% 的宽差，
  且路口和端部以外存在至少 32 m 的连续变化。
- 宽差须超过 `2.5 × max(pixel uncertainty, measurement uncertainty, local width variation) + 0.25 m`。
  局部波动使用宽度及宽差的稳健 MAD，不能通过增加采样数量使其人为缩小。
- 法线 ±5°、互换中心位置、surface/probability 差异提示几何或测量敏感时进入 review。
- 同一 before track 上相距不超过 24 m、至少一个不长于 80 m 的异号变化段，进入交替宽差审核。

`width_precision_candidates` 保存独立 `width_qa_state=accepted/review`；
`width_precision_samples` 保存逐站配对、偏移、置信度与几何检查结果。
全部宽度候选继续保留在原始 diagnostics 和新的 review 图层。道路 Width 主流程未修改。

本轮真实区域对照：

```powershell
runtime/env/samroad_env/python.exe code/tests/run_auto_precision_v2_review.py project/test_area/auto_precision_20260906 project/test_area/auto_precision_v2_20260906
```

此命令只复用缓存候选、存在性观测和两期原 Final 产品，不重新提取道路或读取真值。
报告包括四类前后正式结果、review 原因、典型转待审候选和宽度剖面。宽度审核较保守，
应先检查真实结果再决定是否调松；review 并不是对道路真假或宽度变化真假的最终结论。

道路提取后处理保持原样，本轮不处理提取乱连；Auto 变化组装器也保持冻结。

公共变化字段保持不变；增加辅助审计文件，不改项目 manifest 格式。

### 正式变化几何

`auto_change_geometry` 在 precision qualification 和冻结组装完成后生成正式 polygon。
输入只有已批准对象/seed、原组装 axis/membership/bridge、最终宽度段和已保存的 paired width 剖面；
不接收 Raw Surface、mask、probability 或真值，不重新执行变化判断和连接选择。

- Added 使用 After Final Centerline 与同轨 Final Width；Removed 完全对称使用 Before。
  宽度只从轴线附近 0.75 m 内方向一致的最终宽度段读取，防止取到另一车道。
  无局部宽度的位置沿已有剖面插值；整段缺失时使用该 seed 已记录的宽度，覆盖率写入诊断。
- Widened/Narrowed 沿原 canonical axis，使用保存的有效同符号 paired widths 平滑插值，
  分别生成 After corridor − Before corridor、Before corridor − After corridor。
  无可用剖面时使用原批准 run 的 before/after width，整个变化区间和审核状态保持不变。
- 宽度采用约 4 m 采样、3 点中值与非超调高斯平滑；保留中心线原始折点、平端头和受限转角。
  不用 polygon 平滑移动道路轴，不再与像素路面求交。
- Bridge 沿冻结路径插值相邻 seed 两端宽度；零长度连接不另加方形鼓包。
  组成面只 union 规则走廊，清理不超过 0.01 m² 的数值叠加残片/孔洞，保留真实道路环内部空间。

`published_geometry_parts` 记录正式 seed/bridge 组成面及宽度来源，`published_width_profiles`
记录渲染剖面。原 `local_seeds`、`candidate_audit`、`review_candidates` 和
`network_assembly.gpkg` 中原 `assembly_bridges` 继续用于核对旧证据/连接；正式面为 `changes` / `change_objects`。

只复跑已有真实区域的正式几何（输出目录必须是新目录）：

```powershell
runtime/env/samroad_env/python.exe code/tests/run_auto_geometry_review.py project/test_area/auto_precision_v2_20260906 project/test_area/auto_geometry_20260906_release
```

该脚本逐项比较候选、审核字段及组装连接，输出四类与 junction 对比图。
本区域没有通过审核的 Narrowed 对象，因此其示例明确标为 review-only，不发布为正式变化。

- `road_changes.shp` 和四类变化 shapefile：正式 Auto 结果。
- `auto_diagnostics.gpkg`：正式变化、存在性证据单元、连续未覆盖区间、local seeds、宽度候选段。
- `existence_candidates.csv`：两期 geometry/surface/probability/valid/state/reason 及连续性门限。
- `width_candidates.csv`：样本比例、统计量和宽度候选接受/拒绝原因。
- `candidate_funnel.json`：匹配、存在性、宽度和最终数目的漏斗。
  匹配/证据以约 4 m 单元计数；新增／灭失以 longitudinal interval 和 seed 计数；
  宽度以 axis/run 计数；正式结果以 network object 计数，
  不混用原始 feature 个数。
- `input_provenance.json`：真实输入路径与未使用变化真值的声明。
- `review/`：两期影像缩览、Final 图层、四类叠加、完整 Change Overlay、局部裁图与目录。

渲染已有结果：

```text
runtime/env/samroad_env/python.exe code/tests/render_fast_auto_review.py OUTPUT_DIR SOURCE_PERIODS_JSON
```

`SOURCE_PERIODS_JSON` 仅需 `period_results`，包含期次成果及 `width_review` 路径。
局部图由候选和诊断信息自动选取，不使用变化真值；标签表示检查类别，不是人工确认结论。
若无某类候选，目录明确记为未找到，不伪造示例。

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
  缺少中心线而面/概率仍支持时不判消失。负证据不足则 uncertain。
  有效区和概率有效观测共同约束，边界及 NoData 不自动判变化。
- 只有可靠双向对应且 present→present 的位置允许比较局部宽度。
  复用 paired width measurement/run 组件，在共同法线和各期中心位置测量，
  以配对中点形成输出 canonical axis。
- 默认宽度门限 2 m、20%，并检查有效样本、MAD、方向一致性、配对不确定性和
  栅格分辨率引入的不确定性。输出局部持续变化段，而非整条 feature 平均宽度。
- 默认连续长度 24 m；`min_change_length` 现在实际生效。
  小 gap 只允许在有效、无 junction 且不会跨越反向变化的位置合并。
  宽度候选包含无效样本 gap 时不自动接受。
- 局部检测仍保留 junction 周围 12 m 的原规则；最终结果增加网络级组装，
  根据路口两侧既有同类轨迹补入连接，允许最终变化对象跨 junction 延续。
  详见 `AUTO_CHANGE_NETWORK_ASSEMBLY.md`。匹配有平行轨道歧义或不满足互相对应时不判局部宽度变化。
  极端密集双车道、复杂多叉路口等仍需人工检查，本轮不声称全部解决。

概率读取采用小窗口。地理坐标栅格先将米制采样坐标变换到栅格坐标；
局部 buffer/测宽均在米制 CRS 下计算。连通到整幅影像的道路面先裁到当前轴附近，
再做 union/buffer，防止反复处理整幅面几何。

## 复查输出

公共变化字段保持不变；增加辅助审计文件，不改项目 manifest 格式。

- `road_changes.shp` 和四类变化 shapefile：正式 Auto 结果。
- `auto_diagnostics.gpkg`：正式变化、存在性候选单元、宽度候选段。
- `existence_candidates.csv`：两期 geometry/surface/probability/valid/state/reason 及连续性门限。
- `width_candidates.csv`：样本比例、统计量和宽度候选接受/拒绝原因。
- `candidate_funnel.json`：匹配、存在性、宽度和最终数目的漏斗。
  匹配/存在性以约 4 m 单元计数；宽度以 axis/run 计数；正式结果以连续 run 计数，
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

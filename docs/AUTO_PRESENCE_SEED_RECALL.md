# Fast Auto 新增／灭失候选召回

本阶段只修改 Auto 的 added / removed seed 生成和 QA 发布。
`auto_change_assembly.py` 已冻结；GT-assisted、Final Centerline、Final Road Surface、
道路提取、宽度测量及局部宽度变化检测保持原实现。

## 候选生成

1. 在另一期最终路网的原始折线边上建立空间索引，寻找位置容差内、局部同向
   （方向余弦绝对值至少 0.90）的覆盖。沿本期道路逐边精确相交并转换为纵向区间。
   这一步不依赖 4 m station 的严格 absent 判定，也不受 feature ID 影响。
2. 合并已解释区间并求补集。After 道路的补集产生 added 候选；Before 使用相同函数产生 removed。
3. 4 m 采样继续提供 geometry / surface / probability / valid area 证据。
   对期明确 present（包括已有几何，或面／概率支持）的区间会阻断候选。
   不跨越 present + present 证据补出局部 seed。
4. 相邻 admissible 区间直接合并；confirmed/probable 状态转换和 junction 标记不拆段。
   source present + opposite absent 为 confirmed；source present + opposite uncertain 为 probable；
   source 观测不足为 uncertain。短于最小变化长度或小于最小面积的区间保留为 uncertain。
5. junction 可将 confirmed 降为 probable，并降低规则分数，不硬删除候选。
   新增／灭失几何按最终道路轴和现有宽度形成纵向走廊；不再因道路面局部缺口把候选裁没。

最终道路数据只读。即使观测区无效，几何上两期已匹配的道路仍不会变成新增／灭失。
无效观测不会被解释为 absent，只会作为 probable / uncertain 候选的审核原因。
当前仍无法恢复 Final Road 中完全不存在的道路，也仍保留 present + present 阻断。

## QA 与组装

所有三类 QA seeds 都传入同一个冻结的网络组装函数。
在调用方按 membership 汇总对象 QA，避免只继承首个 seed 的状态：
对象取最低确定性、最低 confidence，保留各 QA seed 数量和 audit_reason 并集。
这一汇总不改变连接选择、轨道或对象几何。

`confidence` 是供排序和复核的规则分数，不是校准后的真实变化概率：
confirmed = 0.90，probable = 0.60，uncertain = 0.30；涉及 junction 再扣 0.05。
宽度结果标记 `paired_width_accepted`，不改变其生成、接受门限和几何。

CLI 参数及公共 SHP 字段不变。对新增／灭失，`min_change_length` / `min_change_area`
本轮改为 QA 降级门限；宽度仍保留原来的硬门限语义。

## 输出与真实区对照

- `auto_diagnostics.gpkg/local_seeds`：所有组装前 seed，含完整 QA、证据比例及 metric axis WKT。
- `auto_diagnostics.gpkg/presence_intervals`：几何未覆盖区间及实际 seed 区间，含拒绝原因。
- `network_assembly.gpkg`：冻结组装的结果、连接和 membership 可追溯轴。
- `candidate_funnel.json`：按方向列出 source axes、未覆盖区间和长度、present 阻断长度、seed 数量与长度、三类 QA 数量和最终对象数量。
- `seed_recall_comparison.gpkg`：旧 local seeds、新 local seeds、最终网络对象。
- `reference_coverage_comparison.json`：事后参考面积覆盖，以及 Final Axis 在参考面内的长度。

可复现实验：

```text
runtime/env/samroad_env/python.exe code/tests/run_auto_seed_recall_review.py BASELINE_DIR OUTPUT_DIR --reference REFERENCE_SHP
```

核心 Auto 函数不接受参考真值。实验脚本先完成 Auto 和矢量写出，再读取参考文件并绘图。
面积 coverage 只反映提供的参考对象，不能代替全区域召回率或 precision 评价。

## 2026-09-06 指定区域实测

复用 `roads_rerun_1788598919` / `roads_rerun_1788599442` 的最终产品，完整重跑 Auto。
结果目录：`project/test_area/auto_seed_recall_20260906`。

| 类型 | 旧 seed | 新 seed | 新 seed 总长度 | 组装对象 | confirmed / probable / uncertain |
|---|---:|---:|---:|---:|---|
| added | 302 | 1207 | 63.39 km | 768 | 9 / 527 / 671 |
| removed | 16 | 363 | 14.31 km | 252 | 0 / 110 / 253 |

所有 1636 个新 seed（含 66 个宽度变化 seed）全部进入组装，输出 1074 个有效对象。
局部拓宽、变窄 seed 的数量、长度和几何与旧结果一致，几何对称差面积为 0。
冻结组装模块、Fast/GT 基线模块和后端调度模块的 SHA256 均未变化。
125 项相关测试通过。

参考对象 0/1/2 的 seed 面积覆盖仍为 0；After Final Centerline 在这些参考面内的轴长也为 0。
对象 3 从 0 提高到 1.35%，裁图显示主要位于交叉道路的路口边缘，尚未恢复参考道路主体。
对象 4 从 10.01% 提高到 18.95%，仍为部分覆盖。组装后这五个对象的面积覆盖与新 seed 相同。
因此只能确认候选供给显著增加，不能据此认为五个真实道路对象的完整召回已明显改善。
本轮未使用真值调参，也未扩展到修改上游道路成果。

旧组装实验从缓存恢复宽度轴，本次正式 Auto 直接传入原始配对轴；
因此宽度组装对象数为 40/14（旧实验 41/13），尽管局部宽度 seed 几何完全相同。
这是组装输入轴的来源不同，未改动组装函数或宽度主流程。

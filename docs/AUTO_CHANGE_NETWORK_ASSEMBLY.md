# Auto 变化结果的网络级组装

本阶段只组装已经检测出的 added / removed / widened / narrowed 局部段。
不重新判断变化真实性，不改变存在性、概率或宽度检测门限，不筛掉任何输入变化段。

调用链：`fast_auto_change.detect_final_road_changes` 完成局部检测后，调用
`auto_change_assembly.assemble_change_objects`，再发布最终对象。
GT-assisted 不调用该模块；道路提取、中心线、路面和宽度主流程均未修改。

## 组装规则

1. 沿最终中心线的交点和端点建立局部路网图。新增/变宽使用 After 路网，
   消失/变窄使用 Before 路网。新检测结果直接保存局部 axis；旧缓存优先使用
   已保存的配对宽度轴，其他段在最终路网上恢复纵向覆盖区间。
2. 同类变化段的端点需朝向彼此，沿路网存在连续路径，且符合道路走向。
   junction 不再是最终对象的终止点：两侧已确认轨迹决定中间补入的道路走廊。
3. 不按 buffer 相交直接合并对象。平行轨道的横向偏移、穿越侧向连接边、
   方向急转会阻止串线；每个端点只选择一条最佳延续。
4. 新增/消失补入沿路径的道路走廊。变宽/变窄按两侧宽度补入纵向条带，
   并与配对轴端部平滑接合，不把整个路口 surface union 当成宽度变化。
5. 每个输入段恰好归属一个输出对象。无法映射或无法连接的段仍保留为独立对象。
   既有段的几何不因组装而裁掉；不根据概率、有效区或参考真值重新筛选输入段。

默认连接距离上限 80 m、端点吸附 1 m、轴投影 4 m、端点方向差 35°、
路径局部方向差 50°，接近笔直的平行轨道横向偏移上限 3 m。
这些是网络组装约束，不是变化真实性门限。

## 真实区域首轮

输入沿用指定的 `roads_rerun_1788598919` / `roads_rerun_1788599442` 的上一轮 Auto 结果。
直接组装已写出的 384 个局部段，没有重跑道路提取或变化检测。

| 类别 | 局部段 | 网络对象 |
|---|---:|---:|
| added | 302 | 260 |
| removed | 16 | 13 |
| widened | 46 | 41 |
| narrowed | 20 | 13 |

共 57 条补入连接，其中 29 条穿过路网 junction。所有 384 个输入段保留且归属唯一。
1 个无法恢复轴的段保持独立。实际区中消失道路的组装样例属于同轨 gap 连接；
消失道路跨 junction 的逻辑与新增一致，具体场景仍需更多实测样例。

参考真值仅在组装与成果写出后读取。提供的 5 个新增真值对象中，
4 个原先没有局部段面积覆盖，另 1 个约 10% 覆盖；这轮组装没有提高这些对象的面积覆盖。
这不是分类精度评价，也不能据此宣称检测召回率提高。仅靠组装无法补出完全没有种子的道路。

## 复查文件

- `network_assembly.gpkg`：`local_change_seeds`、`change_objects`、`object_axes`、`assembly_bridges`。
- 对象层有 `object_id`、`seed_ids`、`seed_count`、`bridge_count`、`junction_count`。
- `assembly_membership.csv`：输入段到网络对象的一对一归属。
- `assembly_decisions.csv`：候选连接的接受/拒绝原因。
- `assembly_summary.json`、`assembly_verification.json`：组装统计及输入保留检查。
- `review/`：整图、四类对象裁图、参考真值事后对照图。

公共 SHP 字段及 CLI 保持不变，组装字段保存在辅助 GeoPackage 中。

```text
runtime/env/samroad_env/python.exe code/tests/run_auto_assembly_review.py SEED_DIR OUTPUT_DIR --reference REFERENCE_SHP
```

`--reference` 只用于事后对照和绘图；核心组装函数不接受真值输入。

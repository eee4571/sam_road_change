# Fast GIS 后处理优化与评价修复（2026-09-08）

本轮只修改主项目；未运行真实项目、模型推理或完整遥感流水线。

## 修改范围

| 文件 | 修改 |
|---|---|
| `code/engine/fast_auto_change.py` | 默认停止 Auto 诊断 GPKG、分类 SHP、候选 CSV、assembly/funnel 调试文件；保留计算和正式变化 SHP、预览、完成摘要。成功发布时清除同目录已知旧诊断文件。离线审查可显式设置 `diagnostics=True`。 |
| 同上 | 概率栅格数据及 mask 估算不超过 128 MiB 时整图缓存；大图使用原始 block、64 MiB LRU、单 block 查询快速路径。整条轴预生成 station/point/normal/cell，共用两期采样坐标并批量查值；宽度、局部道路面查询缓存，避免重复 GeoDataFrame 行访问。 |
| `code/engine/fast_pipeline.py` | 相同输入的期次导出直接复用；验证区只读取一次，每种 CRS 只投影、合并一次。 |
| `code/engine/fast_gt_reconciliation.py` | 没有 cuts/inserts 的 Final Road 建立完成缓存，跳过重复曲面和宽度剖面重建及四类矢量写出。缓存包含输入、中心线属性/几何、CRS、相关代码和输出文件签名。 |
| `code/engine/product_cache.py` | 输入或输出发生变化即失效；只在导出成功后写完成缓存。 |
| `code/engine/road_network_connection.py` | 缓存未变线对的 intersection；无 corridor 重建时复用初始 noding；短环判定复用最短路，接受候选加边后立即清空缓存。 |
| `code/engine/road_track_corridors.py` | 用单位方向向量索引筛选可能满足四度条件的模型对，再执行原始精确判定；保留原始遍历和并列排序。 |
| `code/engine/width/road_change_detection.py` | 评价在投影前后、裁剪前后修复无效面，保留 polygon component、丢弃退化部分；保留原有线评价支持。对象 IoU 使用 STRtree，保留原有 intersection/union、阈值和贪心排序；复用对象匹配，省去逐对象 GeoDataFrame 构造。 |
| `code/temporal_road_analysis.py` | 每期只建立一次新增对象索引，仅查询此前已接受的对象；直接在内存中投影，SHP 只写一次。 |
| `code/engine/fast_timing.py`、`road_network_products.py`、`code/user_pipeline.py` | GIS 阶段 timing。 |

未修改 CLI 参数、4 m station 规则、判定阈值、匹配/宽度规则、GT 后验策略或 Temporal 语义。诊断路径在默认返回结果中为空，正式 `layers` 只列出实际生成的变化图层。必需的道路 GPKG、网络完成审计和 GT/Temporal 内部依赖仍保留。

Fast Width 的区域重建、Export 的连接清理、Final Road 的连续道路面渲染承担不同职责，不能直接互相删除。首次运行保留这些步骤；相同输入的后续 Export 和未编辑 Final Road 复用完成结果。网络变化后的全局 noding 仍保留，以保证远处交点和候选规则一致；本轮优化其重复求交与无变化重复调用。

## 验证

- 语法检查、`git diff --check` 通过。
- 112 项不同的轻量测试通过，包括 Fast Auto、网络连接/产品、Finalization、GT reconciliation、Temporal、局部 CRS、评价和合成期次导出。后续采样和评价调整分别重跑了对应 17 项、18 项测试，均通过。
- 与修改前 Git 代码对照：12 组 Auto 检测的记录、station 审计、类型、属性、几何一致；39 次网络连接结果及审计一致；18 次 Final Road 的四类图层属性及 WKB 完全一致。
- 20 组随机数据（每组 3 期 × 30 条道路）的 Temporal 参考对象顺序、出生期和 WKB 与修改前一致。
- 100 条位于大投影坐标附近的弯曲轴：新旧采样坐标逐元素完全相等。
- RAM/LRU、mask/nodata、越界、缓存驱逐、批量/逐轴概率统计、IoU 并列匹配、缓存输入/输出/属性失效均有回归覆盖。
- 模拟 CRS 转换后产生自交面、混合 GeometryCollection，以及裁剪后仅剩边界线：评价完成且只保留合法面部分。

采样微基准：512×512 合成栅格、2,000 次同区块查询，原实现读窗口 2,000 次，RAM 读窗口 0 次，LRU 读窗口 1 次，所有采样值完全相等。单次本机观测原实现约 0.211 s、RAM 约 0.072 s、LRU 单 block 快速路径约 0.103 s。这些是局部微基准，不是实际项目整体提速比例。

## Timing

日志统一使用 `[Fast timing]`，包含：

- `auto_station_sampling`、`auto_matching`、`auto_assembly`、`auto_total`
- `diagnostics_export_io`、`auto_preview_export_io`
- `regional_network_recovery`、`width_corridor_rebuild`、`period_export`
- `gt_correction`、`final_road`、`finalization_and_temporal`、`finalization_total`、`temporal`
- `evaluation_clipping`、`evaluation_matching`、`evaluation_total`

评价 timing 还写入评价 metadata；完整任务总时间继续使用现有 manifest 的 `elapsed_seconds`。带 total 的阶段包含其子阶段，不能直接相加。缓存命中另报 `period_export_reused` 或 `final_road_reused`。

未验证真实数小时任务的总耗时或其全部输入几何；本轮仅用合成和临时数据验证正确性，不宣称已实测真实项目提速比例。

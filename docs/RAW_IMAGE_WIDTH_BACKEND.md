# 原始影像边界测宽后端

GUI：运行页高级设置 → 宽度提取方法 →「原始影像边界测宽」或「SAM-MoLRA」。默认保持 SAM-MoLRA。选择保存在项目 fast_settings.width_method，重新打开项目恢复；与 IR-MAD、Fast2 四项补偿开关独立。

CLI 新增可选 `--width-method raw_image` / `--width-method sam_molra`，用于 all、extract、extract-project-period、extract-project-all、rerun-period、rerun-all-periods。原参数和默认行为兼容。

## 正式调用链

GUI → task_manager → user_pipeline → 区域最终中心线恢复 → raw_image_backend.measure_region → 原始影像横断面 → Viterbi → 异常识别/连续重建 → width observations → rebuild_network_width_products → 宽度分段/路面 → 原 Fast2 正式入口。

Fast 导出 `export_fast_products` 和 Full 导出 `export_final_products` 均在区域恢复之后接入。它替换最终宽度观测；没有改动上游道路提取和原有为恢复路网准备的辅助产物。SAM-MoLRA 后端仍可选择。GT/Temporal 算法未重构；Fast2 唯一相关调整是宽变配对只使用 A/B 宽度。

正式核心全部位于 `code/engine/width/`：

- raw_boundary_core.py：实验横断面、候选、Viterbi、置信度算法。
- raw_width_reconstruction.py：实验第二层异常识别和连续宽度重建。
- raw_width_records.py：实验观测记录与边界交叉标记。
- raw_image_backend.py：区域输入、缓存、等级映射、现有 GeoDataFrame 接口适配。

实验目录 boundary_width.py / reconstruct_width.py 转调上述正式核心；run_experiment.py 也复用观测记录函数。生产不导入 experiments、tests 或 _profiling。

## 区域计算与数据

使用全部非零长度 degree-2 道路链，沿用全图实验 `max_roads=0, min_length=0`；其余 Config / ReconstructionConfig 参数不变。多影像先建立区域 RGB 输入，再跨 tile 对整条链求解；没有按 tile 分别重建宽度。IR-MAD 输入沿现有 provenance 查回辐射归一化前的影像，原图不覆盖。

缓存记录原影像指纹、最终中心线 CRS/WKB、核心代码和参数配置。导出缓存加入后端选择；切换方法不能复用旧后端最终成果。区域测宽失败会报错，不偷偷回退 SAM-MoLRA。

width observations 保存六个完整字段：final_left_distance、final_right_distance、final_width、final_confidence、width_source、outlier_reason。

等级：measured=A；solver 收敛且 final_confidence≥实验阈值0.16的 smoothed=A；interpolated=B；其余 smoothed、propagated、unresolved=C。C 中有限宽度仍用于显示和道路面，unresolved 不造宽度。Fast2 匹配宽变区间时按最近的实际宽度观测进行 A/B 检查，C 不进入有效宽变证据。

适配层连接共享横断面边界，避免将独立平头 buffer 简单叠加。没有再平滑/重新估计第二层输出的宽度。

内部 observations GPKG / samples.csv 保留详细 QA。正式 Shapefile 字段受10字符限制，使用 final_left、final_righ、final_widt、final_conf、width_sour、outlier_re；发布同一宽度产品的 road_width_profiles.gpkg，保留完整字段名。未发布内部 QA、像素坐标和中间几何编码。

## 实测范围与结果

任务输入：run_20260914_165007，20250118 整区域；复用既有识别缓存，正式 Fast export CLI 完成区域恢复、全区域测宽、重建和导出，再调用正式 detect_final_road_changes 验证 20240918→20250118。前期仍为既有 SAM-MoLRA 产品，用于链路验证，不把混合后端差异当成精度提升。

- 区域输入1792条中心线，恢复后2496条；合并为2336条测量链。
- 67982个宽度观测：A 28056、B 21537、C 18389。
- 来源：measured 8485、smoothed 30496、interpolated 21537、propagated 6916、unresolved 548。
- 67434个道路面，全部 geometry valid；548个 unresolved 未强造路面。
- 正式 export 总墙钟526.58秒，区域测宽/连续重建233.68秒；重复相同 export 命中缓存0.47秒。
- 相邻 Fast2 正式调用111.20秒，宽度配对检查102224个区间查询位置，39119通过 A/B 检查，63105被排除（含缺失/非A/B，不能解读成独立道路数）。未调用 Fast1 baseline。
- 已用正式 ResultPublisher 在验证任务自己的 published 目录验证发布和完整字段 GeoPackage。

结果目录：`project/test_area/_work/tasks/runs/raw_width_backend_validation_20260915/`。产品和预览在 `20250118/products/`，正式发布结构样例在 `published/验证区1/01_单期道路/20250118/`，Fast2 输出在 `20250118/fast2_validation/`。

测试：原实验14项；后端/Fast2/配置/路网41项；user_pipeline + fast_pipeline 165项；随后发布/设置7项均通过。迁移前后15+9个函数/类 AST完全一致。未重跑模型推理、GT后验和完整多期 Temporal；未覆盖原真实项目正式成果。Full 后端接入已做代码/回归检查，真实整期验证使用 Fast 模式。


## 连续道路面输出（2026-09-15）

正式 Fast / Full 导出调用 `engine.width.raw_road_surfaces.build_road_surfaces`：
沿每条 chain 已有共享左右截面拼接完整边界，make_valid 后区域 union，按连通 Polygon 输出。
不重新计算、平均或平滑宽度。异常修复单元保留其有效几何参与融合，缺失宽度不外推。
清理只处理面积小于 1/16 像元的封闭小孔和 1/50 像元边界细节；不做全区 buffer closing、吸附或大范围填充。
独立面不按面积删除，避免删除真实独立道路；保留超过清理尺度的缺口和孔洞。

`surfaces` 为融合面；`width_segments` / `corridors` 保持原始细粒度记录，仅保存在内部任务产品中。
raw_image 后端不再将采样分段 SHP、面片 SHP 或细粒度 profiles GPKG 复制到正式发布目录；
已有公开副本转入内部 publication_history。SAM-MoLRA 发布方式不变。
Fast 输出标记 regular_surface，未受 GT 修改的最终汇总直接使用此面，避免重复对称重建。
导出缓存身份包含新道路面模块；测宽缓存及测宽算法不变。

缓存对比：20250118 的 67,434 个细粒度面片 -> 145 个有效连通道路面，融合约 5.1 秒。
原区域 union 面积 3,221,951.88 m²，融合后 3,221,870.50 m²，净差 -81.37 m²（约 -0.0025%）。
细粒度记录、非几何属性及 WKB 不变；未重跑提取、测宽、GT 或 Temporal。
对比文件：`20250118/surface_dissolve_validation/` 下 comparison.json、road_surfaces.shp、
full_comparison.png、local_comparison.png。原 products 作为修改前基线保留。

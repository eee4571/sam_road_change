# 道路提取与变化检测全流程内存审计

日期：2026-10-12。审计基线：`main` 的 `c7646f8e8f58a18b576d082a977c4743c06ece3b`。

## 结论与范围

**全流程仍存在内存耗尽风险。此前 RGB 的进程内预算不能代表整个软件的 RAM 或显存上限。** 本次是对用户“检查全流程”的补充审计，提交诊断脚本、实际探测结果和本报告；没有更改生产算法、参数、质量分级、文件格式或缓存身份，也没有宣称以下问题已经修复。

实际插件入口是 `plugin/controller.py → plugin/runner.py → code/user_pipeline.py`，GUI 使用 PySide6。`plugin/production_policy.py` 对正式任务固定启用 **Fast、raw_image、IR-MAD**。本次覆盖：输入预检、规范化、IR-MAD、SAMRoad 推理、概率与拓扑、道路面、区域路网、RGB 测宽、正式成果、两期变化、最终跨期协调、长时序、精度评价、导出与预览、预取和续跑。

没有将未启用的旧 MoLRA/GT baseline/Relative 全图分支算成当前默认路径的问题。启用真值辅助或评价后才会进入的分支另行标注。RGB 和上轮修复详见 [RGB_MEMORY_SAFETY.md](RGB_MEMORY_SAFETY.md) 与 [PIPELINE_MEMORY_AUDIT.md](PIPELINE_MEMORY_AUDIT.md)。

## 优先问题

1. **区域路网在 RGB 之前已有无界整线采样。** `road_axis_cleanup._axis_candidate` 对每条非闭合长线生成 1 m 的 `arange`，随后创建逐点 Shapely 插值、坐标、滤波和法线数组。`road_axis_quality` 及后续 `canonical_road_surface`、`auto_change_geometry` 还有同类整线采样。只保护 RGB 无法保护这些入口。
2. **推理阶段保留所有块的 GPU 特征和掩膜。** `inferencer.infer_one_img` 的 `img_features` 和 `img_mask` 在第一遍推理持续追加，到第二遍 TopoNet 才使用，未按字节预算转存或分批释放。当前实际配置是 512 patch、64 margin、batch 16；4096 切片共 121 块，仅三张融合图和保留的双通道 float32 掩膜就为 **434 MiB**，还不含图像特征、模型、batch 激活、CPU 后处理及临时张量。这是内存需求的下界，不是测得的显存峰值。
3. **跨期及跨线程没有总资源准入。** 下一影像推理可与当前影像 CPU 后处理重叠，下一期推理又可与当前期区域恢复/导出/变化分析重叠。队列长度 1 限制的是任务数，不是字节数；独立 Python 进程、Torch、GEOS 和 GDAL 不受同一个总预算控制。
4. **全区矢量及多期成果累计常驻。** 切片完成后仍将四类成果追加到整期 `layer_records`；区域恢复再读入整期四层；长时序同时载入所有期次。高密度道路、复杂面、多期数量会放大 Python 对象、GeoDataFrame、空间索引、GEOS 中间结果和导出副本。
5. **评价分支还有整外包矩形数组。** 通用骨架偏移评价的“800 万像元”计算没有保证最终取整后的行列乘积；狭长范围可突破上限。真值辅助的源网格评价完全按单对象及邻接对象外包矩形栅格化，未设工作格数上限。可能在所有提取和变化计算完成后，最后评价才失败。

## 分阶段检查

| 阶段 / 实际函数 | 已有边界与发现 | 风险判定 |
| --- | --- | --- |
| 输入预检 `user_pipeline._valid_raster_footprint` | 只读 1×1 像元，避免原先全图扫描。但未复用原生解码块检查；源 TIFF 若有巨大条带/压缩块，1×1 请求也不保证底层只解码 1×1。 | 非标准源影像条件风险；源文件元数据应先检查。 |
| 影像规范化 `normalize_validation_sources` | 默认 4096 窗口，通常 RGB uint8 主影像约 48 MiB；还同时存在输入窗口、掩膜、索引赋值副本及 warp 缓冲。`tile_size` 只检查正数，波段数/数据类型可放大单窗；整个当期的所有源 dataset/VRT 同时打开。VRT 虽逻辑覆盖整区，并没有直接申请整区 NumPy mosaic。 | 分窗正确，但输入规模、文件句柄/解码器和全局 GDAL 占用仍无统一预算。 |
| 规范化失败与缓存 | 每窗口落盘及完成状态可复用。但窗口读取的宽泛 `except Exception` 也会接住 MemoryError，随后作为缺少覆盖处理并可能记作 `no_data`，而不是立即以系统资源失败中止。 | 除内存峰值外还有错误分类/续跑完整性风险，应按系统异常明确失败。 |
| IR-MAD `irmad_preprocessing.normalize_pair / irmad_core` | 全量共同像元写磁盘 memmap；迭代最多 400 万 RGB 对，uint8 样本 24 MB，数值计算 CHUNK=262144；不存在全量 float64 N×6 迭代阵列。`windows` 仅限制 128 行，不限制列数；正常规范化输入宽≤4096 时单条带≤524288 像元。memmap 扫描仍会产生映射驻留页/OS 文件缓存，不能把它算作零 RAM。 | 默认规范化路径有实质边界；绕过规范化的超宽输入仍危险；磁盘容量须足够。 |
| SAMRoad 输入 `read_rgb_img / infer_image / rescale_image_to_model_gsd` | 先 OpenCV 全图解码，之后才执行默认 25 MP 检查，过大输入检查过晚。重采样仅检查原图面积，未在 `cv2.resize` 前检查目标尺寸。当前插件 `rescale=off` 且配置为 false，因此放大问题是可选路径，不是当前默认触发。 | 整图解码和可选 GSD 放大需要前置尺寸/字节检查及分块设计。 |
| SAMRoad 模型与 TopoNet | 三张全切片 float32 图 + 所有 patch 特征和掩膜驻 GPU；全图相对对比增强有多张 float32 CPU 图。TopoNet 使用固定 K=16 KDTree 邻居，并非道路点 N×N 全矩阵；但按 batch 内最多节点补齐，节点密集时 `B×Nmax×K×features` 仍大，未按节点总数分批。 | 正常大切片也可能在低显存设备失败，不能只检查文件尺寸。 |
| 概率/道路面 `build_fast_surfaces / _build_fast_road_geometry` | 逐切片 PNG 解码、连通域、骨架和图运算；受上游切片尺寸约束，但缺少本阶段的独立输入上限。骨架像元字典、节点、候选列表属于大量 Python 对象，不能只按 uint8 mask 估算。 | 瓦片较大、噪声/破碎道路多时较高。当前 raw_image 不执行 SAM-MLoRA 推理。 |
| 切片汇总 `measure_fast_widths` | checkpoint 能保存已完成切片；但四层记录全数累计，再写整期 GPKG。复用缓存也会把所有切片数据加载到累计列表。 | 可续跑不等于内存有界；道路数量/顶点数驱动。 |
| 区域路网 `prepare_regional_products / road_network_connection / road_axis_cleanup` | 四层全读、`get_coordinates`、WKB 列表拼接哈希存在额外副本；STRtree 避免普通全局全配对，但相交密集时 cuts/候选仍可很大。1 m 轴线清理在 RGB 前执行。概率点采样的巨大 bbox 已在上轮修复。 | 高优先级：长度预算、几何复杂度及分区处理。 |
| RGB 测宽 | 已有有界采样窗口、道路站点预算、共享工作区/缓存/GDAL 预算、局部 C 级回退及原子检查点。 | 此阶段的内存安全改造有效，但预算只涵盖该进程内受管理资源；不能覆盖上下游。 |
| 正式中心线/宽度段/道路面 `formal_road_products / canonical_road_surface / auto_change_geometry` | 多处按轴长每 1/2/3/4 m 密集采样、逐点构面；GEOS buffer、union、intersection 对顶点/重叠复杂度无预算。 | C 级道路保留后仍可能在这里重新触发大数组。需保留轴线和质量的分段构建，不能随意抽稀后冒充正常结果。 |
| 两期变化 `detect_fast_changes → fast_auto_change → fast_auto_v2` | 证据网格实际≤1200 万格（已修复）；存储两 mask + changed + 三 float32 距离图约 180 MB，另有 SciPy EDT float64/索引临时内存、栅格化 shape 副本。两期场景缓存有数量上限，单读器有 RAM/块缓存上限；但场景中心线、面、宽度段仍全量。`LongitudinalCoverage` 将所有线段展开为独立 GEOS 对象；整期 profiles、buffer、audit 持有到结束。 | 有界栅格仍可占数百 MiB；全区矢量/审计规模无字节上限。 |
| 候选影像核验 `fast_patch_verification.ImageTiles / PatchVerifier` | patch 通常≤192×192，warp_mem_limit=16 MiB。ImageTiles 初始化同时打开两期所有切片和可用面掩膜；该读取器未加原生块检查。 | 小目标窗口不保证源解码/大量句柄有界；不要误报成“192×192 已完全安全”。 |
| 多期证据与最终协调 `fast_multitemporal / fast_gt_reconciliation` | 邻期证据只取需要的邻期；但 `reconcile_periods` 同时读取所有期中心线，发生轨迹修复时全期道路面 `union_all`。文件名包含 GT，但 **无真值的 Auto 最终协调也使用这里**。 | 多期/复杂路网的实际正式路径风险。 |
| 长时序 `temporal_road_analysis.build_temporal_grid` | 所有 period_frames、references、observations、events、life 同时保留；观测记录约按“道路×期数”增长，CSV/DataFrame 构建再复制属性。复核重建还创建完整 expected/actual 关联集合。10 万道路×100 期意味着约千万条观测，不应按两期缓存上限估算。 | 高优先级：按期/道路批量写入，外存索引；人工复核/重建同样受影响。 |
| 真值辅助（有配置时）`_gt_axes` | 紧凑几何拟合失败后，对真值 polygon bbox 栅格化；面积公式+4 边界并非硬性 200 万格上限。 | 条件路径存在与细长证据网格同类漏洞；本次代码确认，未执行巨大分配。 |
| 精度评价（有真值且未禁用时）`road_change_detection` | `_centerline_offset_metrics` 800 万格上限可突破；辅助源网格评价未设格数上限。匹配组构建还枚举预测×真值支持面，完全重叠时邻接集合可能二次增长。 | 已用分配拦截实测确认两类巨大申请；不能当作旧检测器的死代码忽略。 |
| 导出与预览 | 概率拼接已为磁盘+512 窗口；map_scene 掩膜已为目标预览大小。变化预览画布 1100×760，但仍载入全部矢量、遍历所有顶点；正式 SHP/GPKG 输出先建整表，多图层同时持有。 | 巨大 raster mosaic 问题已修复，矢量输出仍非流式。 |
| 插件日志/结果界面 | Controller 历史保留 2000 条，Runner 尾日志有上限；但 `widget.py` 恢复日志用 `read_text().splitlines()[-2000:]`，先读整文件再截尾；Runner 对未换行输出的 `_buffer` 也无字节上限。结果 manifest 整体 JSON 解析。 | 通常低于算法风险，但大日志/大清单存在内存放大；界面没有整个算法的内存预算。 |

## 可执行探测结果

运行：

```powershell
runtime/env/samroad_env/python.exe tests/audit_full_pipeline_memory.py --output _work/full_pipeline_memory_probes.json
```

完整输出随报告保存为 [FULL_PIPELINE_MEMORY_PROBES.json](FULL_PIPELINE_MEMORY_PROBES.json)。这不是“异常处理测试通过”的计数，而是风险诊断。脚本退出 0 仅表示探测运行完成。

| 探测 | 实际代码到达的请求（在分配前拦截） |
| --- | --- |
| 长 10^9 m 的双节点中心线，区域轴清理 | 10 亿个 float64，单个距离向量 **8,000,000,000 B ≈7.45 GiB**。原线本身只需两个节点。 |
| 同一长线，正式变化站点 | 250,000,001 个 float64，单向量 **2,000,000,008 B ≈1.86 GiB**。 |
| IR-MAD 超宽元数据 | 返回 128×1,000,000 条带；两张 RGB uint8 本身 **768 MB**，尚未计 mask、有效像元及 concatenate。没有打开这样大的影像。 |
| 当前配置推理规划 | 4096：121 patch，融合图192 MiB+保留双通道掩膜242 MiB；8192：441 patch，768 MiB+882 MiB。8192 会被当前默认原图25 MP检查拒绝，但拒绝前已经解码；这里仅计算请求量。 |
| 开启 GSD 放大，10 m→0.5 m | 原4096请求 resize 到81920×81920，RGB uint8 **20,132,659,200 B ≈18.75 GiB**；当前配置关闭此选项。 |
| 通用评价，10^12×10 m 狭长面 | “800万格上限”后仍请求 **5×730,296,748 =3,651,483,740 格**。 |
| 辅助评价，大对象 bbox | 请求 **80003×80003**；单 uint8 mask **6,400,480,009 B ≈5.96 GiB**，随后还需另一 mask、骨架与距离图。 |

以上输入是压力/异常几何，不代表真实道路尺寸，也不证明另一台电脑的 40.5 GiB 就出自这些函数。没有实际执行巨量 NumPy、OpenCV 或 rasterize 分配。SAMRoad 仅从实际源码提取纯尺寸规划函数，未加载 Torch 模型或权重。

## 后续改造顺序与约束

1. 首先统一所有几何密集采样的分段预算，覆盖区域恢复（测宽前）、正式构面和变化绘制；保持有效中心线/关联 ID，超预算不可测的宽度保持 C 级，不改变正常道路采样或自动宽变规则。
2. 推理入口在解码/resize 前规划尺寸；为 patch 特征/掩膜提供按字节计的 CPU/磁盘转存及按需载入，为 TopoNet 的节点批次制定预算；需正常数据逐值及 GPU 回归验证，不能只调小缓存或忽略 OOM。
3. 跨线程/进程总预算调度：计入模型常驻、当前后处理、邻期预取、GDAL、场景矢量；资源不足时延后预取，保留顺序执行能力。
4. 整期图层和多期观测改为批量外存累积、分区空间索引与流式导出。GEOS 拓扑分区必须保留跨边界关联，不能为了内存直接切断道路。
5. 补齐所有 raster 入口的原生块检查、句柄上限；修正规范化系统异常分类。评价网格应采用真正格数预算；源网格评价不能随意改分辨率伪造指标，需分区累计/明确未评价状态。
6. 界面日志使用真正的文件尾读取，限制未换行消息累积。

这些是基于证据的待处理事项，尚未完成生产修复。本次不使任何失败阶段返回成功，也不删除缓存以规避问题。

## 验证与未验证边界

- 7 组安全探测全部运行完成，结果见 JSON；其中推理规划含两个尺寸案例。
- 审计脚本 `py_compile` 通过；重新运行 `test_pipeline_memory_safety.py -v`，10 项回归全部通过。这些验证的是此前已修复的拼接/证据/网格读取，不代表本报告的待处理问题已经解决。
- 本次只新增审计文件；没有算法/接口变化，没有重新运行模型，也没有清理或改写推理及项目缓存。
- 未测真实模型 GPU 峰值、完整项目多进程 RSS、千万级矢量峰值、所有 GDAL 驱动或设备。
- 原始失败发生在另一台电脑。精确定位那次 40.5 GiB 申请仍需要完整 traceback（含调用文件和行号）、数组 shape/dtype、当时阶段、影像/切片尺寸、道路规模、内存/显存和运行参数。可以先提供日志和元数据；只有无法用合成数据复现时才需要最小相关缓存或数据。

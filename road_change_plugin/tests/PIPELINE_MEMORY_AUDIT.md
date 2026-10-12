# 测宽之外的内存风险检查

2026-10-12；基于 `main` 的 `3ba255b`。沿实际插件 `fast + raw_image` 路径检查区域恢复、正式产品发布、概率图输出、Fast2 Auto、预览和模型预取。未运行模型；未递归扫描项目成果或环境。

## 结论

**还有风险，不能把上一轮 RGB 的 512 MiB 进程内工作预算视为整个软件的内存上限。** 本轮修复了五组已确认的无界读取/分配问题；正式几何的超长线密集采样、GEOS 整区运算和跨进程叠加仍需独立改造，详见后文。

## 已修复

| 环节与实际入口 | 已确认的问题 | 本轮处理 |
| --- | --- | --- |
| 期次导出：`user_pipeline._write_probability_mosaic` | 同时持有所有概率图与影像掩膜，再分配区域外包范围的概率/有效性两张大数组；稀疏切片间的空白也占内存。该函数实际在 Fast 导出后调用。 | 新 `engine/probability_mosaic.py`：只收集元数据，512×512 窗口读写稀疏 TIFF。非对齐输入经临时磁盘数据和有界 GDAL warp 保持原有重投影。内部 mask、临时目录、原子替换；失败保留既有成果。另限制 TIFF 块目录最多 100 万块，防止巨量目录本身耗尽内存。 |
| 区域路网证据：`road_connection_evidence.RoadProbability.values` | 稀疏采样点构成很大外包矩形；PNG 通过 OpenCV 整幅解码，之后才检查缓存容量。 | 按原生像素位置分组，每窗 ≤512×512；PNG/TIFF 都做有界读取，保持原像素值、mask、重叠取最大值规则。窗口缓存替代整 PNG 缓存。 |
| 轴线修复与变化证据：`road_axis_quality._evidence_route`、`fast_auto_v2.ChangeEvidence` | `sqrt(area / limit)` 未计 ceil/+1，且狭长网格存在最少一行/一列，实际格数可远超宣称上限。 | 新 `bounded_grid.py` 以实际行列数约束 20 万/1200 万格上限；正常未触发上限的分辨率保持不变。仅影响已有允许自适应分辨率的证据网格，不重新缩放或错位匹配影像。 |
| Fast2 概率读取：`fast_auto_change.WindowedProbability` | 逻辑缓存块直接沿用 TIFF 原生条带尺寸，条带过大时单次读就可能失控。 | 超大条带分为逻辑窗口；新增 `raster_memory.py`，原生解码块 >64 MiB 时在读数据前明确报错，因为小 NumPy 窗口不一定能约束底层解码。初始化失败关闭文件，恢复缓存设置。 |
| 预览：`user_pipeline.map_scene` | 预览目标虽然只有约 1400×1000，却调用 `warped.dataset_mask()` 生成源影像全尺寸掩膜。 | 掩膜通过目标预览网格 VRT 读取；其返回数组与预览尺寸一致。读取前检查原生块。 |

上述概率拼接、证据读取、Auto 读取器和预览还复用共享 GDAL 缓存管理，在使用期将原生缓存限制为不超过 32 MiB，最后一个读取器结束后恢复原值。未改变模型权重、推理算法、GUI 命令、概率产品文件路径或波段/掩膜语义。相关区域恢复/正式成果缓存签名补充新工具模块；没有删除推理缓存。

## 仍存在的风险（未声称已解决）

### 高优先级：超长几何的密集站点数组

- `auto_change_geometry._stations()` 按约 4 m 生成整条轴线的站点。
- `canonical_road_surface.py` 的去重剖面、图边、链修复等位置按约 2 m 生成整条轴线站点。
- `road_axis_quality._repair_interval()` 及候选路线重采样按约 1 m 生成站点；异常区间可能很长。**本轮只修复其二维寻路网格上限，不等于修复了后续线性采样。**
- 因此 RGB 阶段已回退 C 的异常长线，在后续正式几何阶段仍可能产生大数组。例如长度 10^9 m 的异常轴线，仅 2 m 间距的 float64 站点向量理论上就约 3.73 GiB，尚未包括 Shapely 点、宽度和质量数组。

这需要统一的几何分段/站点预算，以及跨段拓扑和 C 级质量传播，不能直接随意稀疏化或抛弃长轴线。本轮没有为消除告警而更改这些算法。

### 高优先级：预取与后处理缺少跨进程总预算

`batch_runtime.Worker` 的下一期预取可与当前期区域恢复/导出/变化分析并行。RGB 预算仅在当前 Python 进程内共享，不能限制另一进程、Torch/GPU、累计 GeoDataFrame 或 GEOS。

`samroad.inferencer.infer_one_img()` 按输入切片大小分配三张 float32 融合/计数图到设备，之后还有临时张量和模型特征。正常归一化默认 4096 切片，三张图本身约 192 MiB；绕过归一化、扩大切片或提高批次大小会增大 CPU/GPU 峰值。本轮未执行模型，未验证 GPU 峰值，也未调整推理数值或关闭预取。

### 有界但仍较重：整区矢量及变化证据

Fast2 的网格现有 1200 万格上限；两张 mask、三张距离图和 SciPy 距离变换临时数组仍可占数百 MiB。两期场景还包含概率 RAM/缓存和矢量。`union_all`、全区中心线/面表、空间索引、正式产品合并的成本随对象/顶点数量增长；不会受到 RGB 工作预算约束。

这部分不能仅通过限制数组尺寸证明整机安全。应结合大项目规模和实际可用 RAM，再决定分区、流式发布和预取准入，尤其是低内存电脑。

## 已有明确限制的环节

- 项目影像归一化按 `tile_size` 分窗（默认 4096），不是整区内存拼图；但参数增大仍增大单窗峰值。
- `PatchVerifier.patch()` 的证据图最大约 192×192。
- 场景缓存最多两期；每个概率读取器原有 RAM 128 MiB、块缓存 64 MiB 上限。它们是局部上限，不是全程序总预算。
- `fast_pipeline` 中保留的旧 GT baseline/整图概率函数与实际 `detect_fast_changes → fast_auto_change → fast_auto_v2` 区分检查；本轮没有为旧路径做无关重构。

## 可执行验证

新增 `tests/test_pipeline_memory_safety.py` **10 个测试通过**：

- 概率拼接：对照旧全数组公式，比较所有像素和 mask；覆盖重叠、无效区、非整数对齐、不同分辨率、旋转以及禁用直接放置，全部逐值相等。
- 80032×80032 的稀疏区域（仅两张 32×32 源图）：禁止 >512² 的 NumPy 零数组申请，输出 <10 MiB，空白区无效，远端像素正确。旧双数组理论上约 11.93 GiB，本测试没有尝试该分配。
- 发布中注入 PermissionError，既有成果字节不变、临时目录清理。
- 过大 TIFF 目录在创建输出前拒绝。
- 两个相距约 8 万像素的证据点，实际只读两个 1×1 窗口，数值和越界 NaN 正确。
- PNG 路径禁止 OpenCV 全图解码，采样值正确。
- 极端长宽比、取整边界及实际 ChangeEvidence 构造均不超过单元预算。
- 原生巨大解码块在读取前拒绝；Auto 文件句柄和 GDAL 设置恢复。
- 60 万列的条带概率图使用小逻辑窗口，采样值正确。

相关回归全部通过：`check_processing_optimizations.py` 25、`check_formal_period_products.py` 17、`check_fast2_structure.py` 23、`test_width_local_exceptions.py` 15、`test_ui_processing.py` 35。连同新增 10 项，本轮 **125 项通过**。本轮没有把上一轮未重新执行的测试计入数量。

已有项目 `plugtest` 缓存检查：44 条输入、55 条恢复线、4 条测宽链、141 个断面；与上轮 `period-20221020-54807a8c` 的总宽、左右距离、来源、质量和几何逐值一致。输出 `project/plugtest/_work/_processing_check/period-20221020-0e0e86e5/`。无模型推理、无正式成果覆盖。

```powershell
runtime/env/samroad_env/python.exe tests/test_pipeline_memory_safety.py -v
runtime/env/samroad_env/python.exe tests/check_processing_optimizations.py -v
runtime/env/samroad_env/python.exe tests/check_fast2_structure.py -v
runtime/env/samroad_env/python.exe tests/check_existing_period.py ../project/plugtest
```

此报告是限定实际调用链的代码检查和合成/缓存验证，不是所有输入、所有模型环境或整个应用 RSS 的无条件安全保证。

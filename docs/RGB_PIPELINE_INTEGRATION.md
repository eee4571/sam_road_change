# 独立双实现：验证区网格、IR-MAD 与 RGB 测宽

工作台和 QWidget 插件分别维护自己的 `code/`、配置及运行环境，不互相 import，也没有运行时同步器或公共 engine。插件的 `backend_snapshot.json` 仅记录插件自身文件校验值。

## 当前正式链

验证区统一网格/tiles → Fast IR-MAD → SAMRoad/TopoNet → Fast Mask/中心线后处理 → 区域路网恢复 → RGB影像边界测宽 → 路口间代表性宽度 → 规则道路面 → 写出成果 → Fast2 → 原 GT 后验/Final/Temporal 链。

- 工作台 Step1 按验证区选择“IR-MAD参考期”；插件数据页选择验证区和参考期。选项来自该区期次，保存于项目 `area_irmad_references`。
- CLI `--irmad-reference` 支持单期名称或 JSON 区域映射。任务 `input_spec.irmad.reference_period` 保存该选择，续跑及局部重跑读取任务已保存的参考期。在项目中改变参考期后，用相同任务完整续跑会按受影响区域的输入身份失效并重建。
- 每个目标期独立与所选参考期拟合；参考期仅经过统一网格，不做自归一化。仍为最多 4M 确定性迭代抽样，全量共同有效像元计算最终 NCP/PIF/TLS。IR-MAD 本身不重新采样。
- 数据检查报告影像范围覆盖率；统一网格记录实际有效像元覆盖率及缺失像元，缺失位置保留 NoData。
- `raw_image` 完全跳过 MoLRA 模型、缓存概率及旧横断面测宽。测宽读取实际推理使用的归一化 RGB tiles，参考期读取其统一网格原影像。
- tile 阶段名仍用内部 `width` 键保持调度兼容，但 RGB 分支只生成后处理中心线和 Fast mask；名义宽度只用于内部拓扑准备，不是测量或 Fast2 输入。
- 独立 `regional` 阶段完成恢复、RGB 测宽及道路产品构造。`export` 只序列化并生成预览。旧阶段缓存以 RGB 管线版本和网络产品版本失效。
- 每条 degree-2 连续 chain（止于 junction/endpoint）使用可靠 A/B 观测的长度加权中位左右距离。没有可靠观测的 chain 仅可保留 C 级显示宽度，不成为 Fast2 宽变证据。逐点连续测量不被改写。
- Fast2 和正式道路面使用同一代表性宽度；逐点 profile、细粒度观测保存在内部 `width_observations.gpkg`。宽度段/corridor 使用内部 GPKG，不再写无用 SHP。
- 预取仍是单 GPU worker；`next_period_preparation` 和 `next_period_prefetch` 单独计时，区域构造计入 `regional_products`，写出计入 `period_export`。

## RGB 特征缓存的精确性边界

块级 RGB/Lab/灰度缓存与完整窗口的 Gaussian/Sobel/Canny/texture 缓存共享 128 MiB LRU。各 road chain 的 Viterbi 和连续重建使用最多 4 个线程，顺序收集结果。tiles 按窗口读取，不写整区 RGB mosaic。

Canny 滞后连接依赖完整窗口，不存在固定小 halo 能严格保证分块结果等价。因此非局部滤波保留原 crop 边界，并按完整窗口复用，不将不同 crop 的 Canny 结果近似为同一个块结果。超预算窗口照常计算但不驻留。spacing、cross_step、候选数、Viterbi 和连续重建参数不变。

## 验证范围

仅小型合成影像、mock 模型、配置/缓存/重跑/发布及 GUI 启动测试；不运行真实多期模型任务、不做参数调优。新增测试覆盖跨 tile 读取、精确特征复用、并行输出顺序、RGB 分支不进入 MoLRA/旧测宽、代表性宽度、导出不触发重建、分区参考期及缓存身份。

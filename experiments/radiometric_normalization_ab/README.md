# 独立 IR-MAD 辐射归一化实验

当前固定 `T1=20250118`，只将 `T2=20260203` 归一化到 T1。复用 A 中已有 T1、Raw T2 道路成果，只运行 Normalized T2 提取。**不运行 Fast2、变化检测、GT 校正、Temporal 或多期批处理。** 不写正式代码、原始影像或正式成果。

## 方法

- 全部共同有效 RGB 像素做 IR-MAD，最多 30 次；CCA 相关系数最大变化小于 0.01 时收敛，与 ArrNorm 默认设置一致。
- `NCP > 0.95` 选择 PIF；逐波段正交 / TLS 回归，`reference = gain * target + offset`。所有瓦片共用参数，不链式归一化，不使用道路、GT、Fast2 结果选择 PIF。
- 原始 TIFF 不同 CRS/分辨率；复用 baseline 已有的 8 对严格共网格、未做辐射归一化的分析瓦片，零新增 warp。推理网格与 baseline 完全相同。
- 另存原始 T2 网格的 normalized TIFF 用于 GIS；不将它再次空间处理后投入推理，以避免二次处理影响对照。
- 保持 CRS、transform、shape、dtype、nodata、逐波段有效性、波段说明及非统计元数据。采用无损 DEFLATE；删除失效 DN 统计标签。uint8 四舍五入并裁剪至有效编码范围；原生 TIFF 的 255 保留为 nodata，记录裁剪数量。不改模型内部 normalization。
- 使用 baseline 冻结生产代码 `snapshot/code`，提交 `734fe17cada16effc181336bfe1b18ad5e75daad`，相同模型、配置、fast 提取步骤和 Auto 参数。Auto 因影像改变 profile 属于原流程行为。

参考 [SMByC/ArrNorm](https://github.com/SMByC/ArrNorm)、[IR-MAD](https://github.com/SMByC/ArrNorm/blob/master/core/iMad.py) 和 [正交回归](https://github.com/SMByC/ArrNorm/blob/master/core/auxil/auxil.py)。使用 NumPy/SciPy 独立实现 SVD CCA、全像素流式矩和 TLS，未复制或安装开源程序。

## 执行

在仓库根目录执行。依赖本地已有 A 成果、冻结代码、模型及实验配置，Git 只保存代码和说明，不含这些资源。已完成步骤有防重复检查。

```powershell
runtime/env/samroad_env/python.exe -B experiments/radiometric_normalization_ab/run_irmad_experiment.py normalize
runtime/env/samroad_env/python.exe -B experiments/radiometric_normalization_ab/run_irmad_experiment.py extract
runtime/env/samroad_env/python.exe -B experiments/radiometric_normalization_ab/compare_irmad_roads.py
runtime/env/samroad_env/python.exe -B experiments/radiometric_normalization_ab/report_irmad.py
runtime/env/samroad_env/python.exe -B -m unittest discover -s experiments/radiometric_normalization_ab -p 'test_irmad*.py'
```

`finish_irmad_outputs.py` 仅用于恢复本次中断的 TIFF 导出；不拟合参数或推理，保留失败文件，逐像元验证已有输出。

## 本地输出

当前新增 **原始 TopoNet 展示**：运行 `show_raw_toponet.py`，直接读取三组已有 `*_fast_topology.npz`，绕过全部下游道路后处理，不重新推理。结果在 `irmad/raw_toponet/`，包括独立显示、叠加 T1 对照、全区图和保留置信度的 `raw_toponet.gpkg`。保留原模型候选点生成和连接阈值 0.5；不补线、删支、平滑或进行道路面约束。此前 `irmad/evaluation/` 的图和数值仍是后处理成果，保留追溯，不能与原始 TopoNet 图混淆。

- `irmad/normalization.json`：参数、迭代、PIF 分布、裁剪和保真检查。
- `irmad/normalized_tiles/`：实际推理用的 T2 GeoTIFF。
- `irmad/normalized_native/20260203.tif`：原始 T2 网格的归一化 TIFF。
- `irmad/pif/`：NCP 图，-1 为非共同有效像元。
- `irmad/T2/`：唯一新增道路成果、SAM-MoLRA 概率、道路面和自然生成的宽度。
- `irmad/evaluation/`：采样点、稳定道路宽差、指标、图和 `REPORT.md`。
- `irmad/*audit.json`：原始副本和 baseline 的哈希保真检查。
- `irmad/*.log`：本次日志；运行时缓存和临时文件仍在本实验 `cache/`、`tmp/`。

评价使用验证范围内的共同有效像素；中心线每约 2m 采样，3m 距离和 30° 方向阈值匹配，另报 1/5m 敏感性。稳定道路是两种 T2 均匹配的长 T1 道路代理，不是真值。恢复、丢失、未匹配和新生片段不能直接解释为准确率或假变化数量。分别比较覆盖、偏移、道路总量、原始/增强 SAM-MoLRA 面、最终道路面，以及注明 fallback 比例的自然宽度。

## 历史成果与 Git

`A/`、`B/` 和旧 `run_experiment.py`、`evaluate.py`、`finish_experiment.py`、`frozen_fast2.py` 等属于此前 median/IQR 完整 A/B，保留追溯。**本轮不再调用旧入口**，只复用工具函数、A 成果和冻结配置。旧评价未完成，不能视为 IR-MAD 结论。

`.gitignore` 默认忽略全部生成物，只允许根目录 Python、README、gitignore 和 bootstrap Python。影像、配置中的本机路径、模型、日志、缓存、图表和报告均不上传。

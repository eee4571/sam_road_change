# 正式 IR-MAD 影像预处理

GUI 高级设置提供 **跨时相辐射归一化（IR-MAD）**，默认关闭，参考期固定为 **20250118**。

CLI `all`、`extract-project-period`、`extract-project-all` 支持 `--irmad` / `--no-irmad`。命令行续跑不指定开关时沿用原任务设置；GUI 使用当前复选框的明确选择。期次局部重跑沿用任务保存的设置。

流程：原有分析网格准备 → 可选 IR-MAD → 道路提取 → 原有变化检测、GT 后验和 Temporal。IR-MAD 本身不重投影、不插值、不增加几何重采样。它保留输入分析瓦片的 CRS、transform、shape、dtype、NoData 与有效 mask；原有验证区网格准备仍按原流程执行。

## 固定参考与独立拟合

- 每个区域必须有 `20250118`，参考期直接使用未做辐射归一化的分析影像。
- 每个其他期次与原始参考期独立计算 IR-MAD、PIF 和 TLS；不使用上一目标期的 PIF、参数或 normalized 影像。
- 与实验一致：一个目标期的全部配对瓦片共同参与该期的全像元统计，期内共享其 TLS 参数。不同目标期独立拟合。
- 瓦片必须同名、严格共网格，且为三个 uint8 RGB 波段。不满足条件时报错，不隐式转换或回退原图。

## 与实验一致的算法

`code/engine/irmad_core.py` 原样迁入实验中的数学及 TIFF 写出函数；回归测试逐函数比较 AST，防止改变算法。

- 全共同有效像元，float64 流式矩，chunk=262144；SVD CCA。
- 最多 30 次迭代，相关系数变化收敛阈值 0.01。
- NCP > 0.95 为 PIF；每波段 orthogonal/TLS：reference = gain × target + offset。
- 保留实验未收敛时选择最小 delta 迭代的处理，并在 audit 记录 `converged=false`；没有改成另一种模型。
- uint8 四舍五入、按有效编码范围裁剪、无损 DEFLATE，保留原无效像元与 mask，逐块核对网格及 NoData。

不足共同像元、退化协方差、无效 TLS、网格不一致、读取或写出异常均明确报错。失败产物不能作为完整缓存。

## 缓存和续跑

项目 `_work/cache/irmad/<identity>/` 单独存放 normalized_tiles、PIF/NCP 图、独立拟合 audit 和完成标记；原影像不覆盖。身份包含启停、参考期、源/参考影像文件指纹、分析网格依赖、算法版本、IR-MAD/PIF/TLS 参数及输出配置。

开启时模型工作目录位于对应期次的 `radiometric/<identity>/`，避免旧 Raw 模型中间文件被 `resume` 或 GPU 预取复用。关闭使用原有 Raw 路径。切换模式或源/参考输入变化时重建受影响道路与下游成果；同配置续跑复用已经完整的成果。

正式变化和道路成果目录不变。新增身份及归一化审计保留在内部 input/period/task manifest。Fast2 的四项补偿配置不随此开关自动改变，GT-assisted 和 Temporal 算法也不改变。

验证使用三期合成 GeoTIFF 和调度桩：参考期零拟合，两目标期独立拟合；输出网格/逐波段 mask/NoData/原文件哈希不变；开关、缓存命中与失效、续跑、GPU 预取输入、任务复制后的缓存路径和错误传播。未运行真实区域模型推理。

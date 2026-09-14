# Fast2 可选跨时相补偿

配置入口位于 `engine.fast2_compensation`。默认 `raw_input`，保持拆分前的 Fast2 运算及参数；此配置不影响候选生成、correspondence、几何重建和发布分级的实现。

```python
from engine.fast2_compensation import Fast2CompensationConfig
from engine.fast_pipeline import detect_fast_changes

raw = Fast2CompensationConfig.from_preset('raw_input')        # 四项 True
normalized = Fast2CompensationConfig.from_preset('normalized_input')  # 四项 False
custom = Fast2CompensationConfig.from_preset(
    'custom', patch_radiometric_normalization=False,
    width_temporal_bias_correction=False,
)  # 未指定项沿用 raw_input

result = detect_fast_changes(before_result, after_result, output_dir,
                             compensation=custom)
```

也可以直接构造 `Fast2CompensationConfig(...)`。配置不可变，只接受真正的 boolean，不将字符串 `"false"` 当成布尔值。`raw_input`、`normalized_input` 不允许混用逐项覆盖；覆盖应使用 `custom`。

CLI 的 `all`、`change`、`change-project-periods` 支持以下可选参数，其他参数和 GUI 原调用保持兼容：

```text
--fast2-compensation normalized_input
--fast2-compensation '{"preset":"custom","patch_radiometric_normalization":false}'
```

`all --resume` 不指定此参数时沿用原任务配置；显式改变 preset/开关，仅使变化和下游 Final/Temporal 缓存失效，不重新提取已完成道路。期次/变化对局部重跑、批量重跑沿用 manifest 的 `input_spec.fast2_compensation`；独立变化对沿用其结果中的配置。

## 四个接口

| 模块 | 输入 → 输出 | 关闭行为 |
|---|---|---|
| `PatchRadiometricNormalization.grayscale` / `apply` | RGB → float32 灰度与有效 mask；已配准灰度与 background mask → 灰度、gain/offset | 不做 patch 分位数拉伸，不改变有效灰度的辐射尺度；gain=1、offset=0。NaN 的 CV 表示处理保留 |
| `StableAppearanceCalibration.fit` | 原有稳定道路结构描述 → 同字段的参考统计 | 不估计跨期分布，返回 identity 参考：差异与偏差为 0、相似度为 1、spread 为 0；真实有效控制数量保留 |
| `SurfaceProbabilityCalibration.apply` | probability、surface 数组 → 校准后数组 | 原数组原样返回 |
| `WidthTemporalBiasCorrection.fit` | 可靠稳定道路的宽差 → bias/scatter 与审核信息 | bias=scatter=0，已有 width profile 不变 |

**当前 Fast2 没有 probability/surface 的跨期全局校准算法。** 因此第三个接口开启时也执行 identity，明确记录 `identity_no_existing_cross_period_correction`。此次不补造校准算法。单期 probability percentile/rank、surface 存在性证据全部保留，不把这些证据当作跨期补偿禁用。

局部几何配准仍在 `align_pair` 执行；其是否接受平移仍由原安全判断决定。原始梯度、方向纹理、NCC/SSIM、道路两侧边界的计算，以及 `strong_stable / strong_change / uncertain` 和四级发布裁决均保留。关闭外观校准仍读取稳定控制 patch、保留有效性检查；不足的有效证据仍产生 uncertain，不伪造有效控制。

`normalized_input` 是后续消融实验接口，不是本轮重新调出的算法：开关会改变送往原判定逻辑的补偿值，因此不保证它与 `raw_input` 发布相同数量。没有引入 IR-MAD 或新增特征。

## Diagnostics 和缓存

- `patch_verification.json.compensation`：四项实际开关、实现版本 identity，以及第三模块的 identity 状态。
- 内部 Auto summary / manifest：`fast2_compensation`、`fast2_compensation_identity`。
- 直接 `analyze_scenes` 的返回统计：`compensation`、`compensation_identity`。
- identity 包含四个开关与补偿实现版本。不同开关不复用变化缓存。未标注开关的历史当前版本 Fast2 缓存仅允许用于 `raw_input`。
- 道路场景的只读栅格/索引缓存未被补偿原地修改，可以继续复用。正式 Shapefile 不增加补偿或来源字段。

## 验证

`code/tests/run_fast2_compensation_ab.py` 在改造前捕获合成道路、真实 GeoTIFF patch 读取和结构特征结果，改造后逐字段比较候选、presence/width/image audit、非计时统计、浮点字节及几何 WKB。只排除计时和新增补偿配置元数据，不使用容差或 rounding。

`code/tests/test_fast2_compensation.py` 覆盖独立开关、identity、关闭补偿仍配准/计算结构、正式入口诊断、版本隔离，以及全任务配置切换/续跑缓存。

## GUI 设置

运行页的「变化检测」区域，点击「Fast2 / 变化检测高级设置」：四个中文复选框分别控制现有四项补偿，默认全开，项目重开后恢复。IR-MAD 开关及参考期独立保存，不自动联动。完整运行、单对和批量变化重跑均传递当前补偿配置；CLI 不提供覆盖参数时仍沿用原任务配置。日志记录 `[Fast2 settings]`，diagnostics 和缓存身份继续使用已有配置字段。

「道路面与概率跨期校正」仍连接现有接口；该接口当前为 identity（尚无独立整体校正实现），本轮没有为使开关产生差异而添加新算法。

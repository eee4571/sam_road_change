# 原始影像道路边界测宽实验

实验入口调用正式 `code/engine/width/raw_boundary_core.py` 与 `raw_width_reconstruction.py`。算法只维护一份；本目录保留实验运行、可视化和对比工具。

## 实际结果

### 第二层：全图连续宽度重建

- [连续面矢量叠加前后 PNG](results/continuous_20250118_v2/continuous_overview_comparison.png)
- [原始分辨率连续面 PNG](results/continuous_20250118_v2/continuous_overlay_native.png)
- [连续宽度面和左右边界矢量](results/continuous_20250118_v2/continuous_surfaces.gpkg)
- [五类 profile 对比](results/continuous_20250118_v2/profile_comparison.png)
- [来源图](results/continuous_20250118_v2/width_source_map.png)、[实测报告及失败案例](results/continuous_20250118_v2/REPORT.md)

本层在保存的 `samples.csv` 上运行，保留全部原始字段；不改变下述第一层的候选生成、Viterbi 和拒绝规则。全图 4,346 条链、76,863 点的重建覆盖率从 34.86% 到 91.33%；其余链至少一侧完全没有可靠支撑，保留空值。覆盖率不表示准确率，插值和传播不算新增实测证据。

复现（输出目录必须尚不存在）：

```powershell
& runtime/env/samroad_env/python.exe experiments/raw_image_boundary_width/reconstruct_width.py experiments/raw_image_boundary_width/results/full_image_20250118 experiments/raw_image_boundary_width/results/continuous_new
& runtime/env/samroad_env/python.exe experiments/raw_image_boundary_width/render_reconstruction.py experiments/raw_image_boundary_width/results/continuous_new
& runtime/env/samroad_env/python.exe -m unittest discover -s experiments/raw_image_boundary_width -p 'test_*.py' -v
```

`reconstruct_width.py` 的 `ReconstructionConfig` 保存所有重建参数，写入新 manifest，并记录原观测 CSV 的 SHA-256。算法依次为：

1. 左右侧分别寻找最长 30 m 的返回型异常，比较前后各 15 m 内至少 3 个支持点。两端中位数差须 ≤1.5 m、各自 MAD ≤0.9 m；中间至少 80% 的可用点同向偏离两端趋势超过 `max(2.5 m, 3×1.4826×MAD)`。既检测变宽也检测变窄，即使两侧相反误差使总宽不变也能检测。持续台阶和渐变没有返回两端，不按此规则删除。
2. 对置信度 <0.20 的观测，检查 ±24 m 内可靠邻居支持的局部线性趋势；偏离超过 `max(3 m, 4×1.4826×残差MAD)` 的点标记趋势冲突。路口上述判异阈值加倍。影像边界、搜索端点、无状态和几何交叉观测不能作为锚点。
3. 原始 accepted 点及置信度 ≥0.16 的非硬拒绝路口单侧观测作为锚点，删除该侧异常后独立拟合。优化目标为置信度加权 Huber 观测项（转折 1 m）+ 0.35×Huber 一阶距离差 + 2.5×Huber 二阶距离差；空间差按实际采样距离归一到 3 m，转折分别 0.6/0.4 m。L-BFGS-B 约束每侧 0.75–20 m，记录收敛状态；没有定宽目标或左右对称约束。
4. 路口连续约束降到 0.25 倍，观测拟合权重降到 0.20 倍。这样允许快速变化，也避免把第一层原先拒绝的路口边缘当作强证据。
5. 两锚点间 ≤18 m 缺口用平滑三次过渡，18–75 m 用保形三次插值恢复局部趋势；>75 m 用两端各 18 m 稳定邻域中位数保持并在中间过渡。链端采用附近稳定距离延续，避免无限延长斜率。这些曲线仅作较弱先验（权重 0.12），再与观测一起稳健优化。没有锚点的一侧不引入全图平均宽度。
6. `final_width = final_left_distance + final_right_distance`；按存储法线重建最终左右边界坐标。插值置信度取两端较低值并衰减；传播置信度不超过 0.20，随距最近锚点距离按 120 m 尺度衰减；路口另乘 0.65。分数未经精度标定。

新增字段：`final_left_distance`、`final_right_distance`、`final_width`、`final_left_x/y`、`final_right_x/y`、`final_confidence`、`width_source`、`outlier_reason`、`reconstruction_available`、`reconstruction_flags`、`solver_converged`；另有逐侧来源、锚点、异常原因和置信度。最终来源取两侧较弱的一类：

| width_source | 含义 |
|---|---|
| measured | 原始有效观测，两侧各改动 ≤0.25 m |
| smoothed | 观测稳健拟合，包含重新利用的路口观测 |
| interpolated | 两端有支撑的短/中缺口 |
| propagated | 长缺口或链端的稳定值延续，降低置信度 |
| unresolved | 至少一侧整链无可靠锚点，最终宽度空值 |

输出保留 CSV/JSON 原始观测与最终值、逐道路摘要、异常连续段清单、全图连续面/边界 GPKG、原生 PNG、来源图、五组真实影像与 profile 对比。面由最终左右横断面连接；交叉四边形拆分为合法面并降置信度，自交边界另行标记。不能把几何合法性当成边界准确性。

实际结果：判异常 5,147 点 / 2,280 段；同一原始有效相邻点对的平均跳变 0.633→0.390 m，P95 2.000→1.272 m。所有可比较有限观测（含原始拒绝值）的相同点对平均跳变 4.603→0.709 m。短时异常下降，但持续错误边缘、中心线落在树带、长缺口推断仍是失败来源。短小真实停车湾也可能被返回型判据误删；无真值时不能声称精度或 SAM-MoLRA 替代性已获验证。

### 第一层：原始影像观测

全图面矢量成果：

- [测宽面矢量叠加 PNG](results/full_image_20250118/width_surface_overview.png)
- [原始分辨率叠加 PNG](results/full_image_20250118/width_surface_overlay_native.png)
- [面矢量 GeoPackage](results/full_image_20250118/width_surface_polygons.gpkg)
- [局部放大对比 PNG](results/full_image_20250118/width_surface_details.png)

全图处理取消道路数量及70m长度门槛，共4,346条道路链。面由相邻有效横断面的左右边界直接连接，保留连续宽度变化，不使用固定宽度中心线缓冲，不填补低置信缺口。PNG直接读取导出的面矢量进行透明填充叠加，交付不依赖网页。

前次120条较长道路实验报告在 [results/real_20250118_final/REPORT.md](results/real_20250118_final/REPORT.md)，包含五类主要案例及失败案例。`pilot`、`real_20250118`、`real_20250118_v2` 是开发期间先导结果。

`.gitignore` 仅允许本实验源码、测试、`cases.json` 和 README 进入版本控制；原始影像、PNG、矢量、CSV/JSON/NPZ诊断和结果目录均保留本地。

全图复现（使用新的输出目录）：

```powershell
& runtime/env/samroad_env/python.exe experiments/raw_image_boundary_width/run_experiment.py --max-roads 0 --min-length 0 --skip-road-plots --output experiments/raw_image_boundary_width/results/full_new
& runtime/env/samroad_env/python.exe experiments/raw_image_boundary_width/export_width_surfaces.py experiments/raw_image_boundary_width/results/full_new
```

面矢量 `width_surfaces` 图层使用米制UTM坐标，包含道路ID、起止样点/里程、左右距离、平均宽度、起止宽度、置信度和面积。面片平均宽度为相邻两个横断面宽度的均值。

## 运行

在仓库根目录使用现有环境（无需模型/GPU）：

```powershell
& runtime/env/samroad_env/python.exe experiments/raw_image_boundary_width/run_experiment.py --output experiments/raw_image_boundary_width/results/new_run
& runtime/env/samroad_env/python.exe experiments/raw_image_boundary_width/review_cases.py experiments/raw_image_boundary_width/results/new_run --cases experiments/raw_image_boundary_width/cases.json
& runtime/env/samroad_env/python.exe experiments/raw_image_boundary_width/build_report.py experiments/raw_image_boundary_width/results/new_run
& runtime/env/samroad_env/python.exe -m unittest discover -s experiments/raw_image_boundary_width -p test_boundary_width.py -v
```

`--image`、`--centerlines`、`--layer` 可更换输入；当前限定有 CRS 的 uint8 RGB 三波段影像与线矢量。`--max-roads 0` 处理全部符合长度条件的道路链；默认选择最长的 120 条。输出目录必须不存在，防止覆盖既有实验。`cases.json` 的道路编号和里程只适用于默认真实数据；换输入需重新选例。

依赖：NumPy、SciPy、OpenCV、Rasterio、GeoPandas、Shapely、PyProj、Pandas、Matplotlib、Pillow。未新增环境依赖。

## 算法

1. 将中心线投影到按区域估计的 UTM 米制坐标。去重反向重复预测边，节点化真实几何交点，合并 degree=2 链；不进行补线、道路面约束或模型推理。
2. 沿链先约 1 m 重采样，Gaussian 平滑 sigma=2 m，保留端点，再按弧长每 3 m 采样。用前后各 5 m 的弦方向估计切线，法线定义为行进方向左侧。
3. 在原始影像中双线性提取法线横断面：候选搜索 ±20 m，外侧再读 2 m 证据缓冲（实际 RGB/Lab 保存 ±22 m），横向步长 0.5 m。缓冲保证搜索顶端也有两侧色差证据，并能正确触发 search_limit 标记。像素中心采用 affine inverse 减 0.5；不把角度单位当米。按每条道路包围框窗口读取，未重写原图。
4. 四项响应均固定尺度裁剪到 [0,1]，权重依次为 0.35/0.25/0.15/0.25：
   - Lab 两侧 0.5–2 m 色差；
   - Lab 横断面梯度；
   - 7×7 像素灰度局部标准差的横向变化；
   - 灰度 Sobel 幅值及 Canny 边缘。
   色差/梯度/纹理/Sobel 归一尺度分别为 22、12、0.07、0.10。不是每条横断面独立拉满响应，平坦区域不会被强制制造高响应。
5. 每侧保留最多 7 个局部峰，峰间至少约 1 m，包含可用搜索端点；枚举联合左右状态。左右距离各至少 0.75 m（离散网格实际 1 m），宽度限定 2–36 m。
6. **基线**在同一候选集合及同一几何约束下，每个横断面独立最大化左右响应之和。
7. **连续方法**采用联合左右状态的 Viterbi 动态规划，精确求解给定候选集合上的一阶能量：

   `E = -Σ(response_left + response_right)`

   `    + 0.35 Σ[Huber(Δleft/Δs) + Huber(Δright/Δs)]`

   `    + 0.50 Σ Huber(Δwidth/Δs)`

   Huber 转折点为 1。无左右对称项、无固定宽度目标；缓慢拓宽或变窄只产生小惩罚，强响应可支持真实变化。`width = left_distance + right_distance`。候选截断意味着这不是全像素状态空间的全局最优。
8. 两侧正距离保证单横断面左右不交叉；额外检查每条有效链的左右轨迹相交或自交，将发生几何冲突的整段降为诊断输出。

## 置信度和断链

边界置信度为 `strength × (0.25 + 0.75 × uniqueness)`；uniqueness 比较同侧相距至少 2 m 的最强竞争响应，差值除以 0.3 后裁剪。它是**未标定的影像证据分数，不是准确概率**。

- 图上 degree≥3 的节点周围 12 m 标记 `junction`；没有预测出来的支路不会被自动识别为路口。
- 含外侧证据缓冲的 ±22 m 横断面含无效像素、越界、nodata 时标记 `image_or_nodata_boundary`。这是保守策略，即使道路本身可见，搜索窗口越界也会拒绝。
- 任意一侧置信度 <0.16 标记 `ambiguous_or_weak`；优化后选中的弱边界标记 `optimized_weak`，重新分段优化直到屏蔽集合稳定。
- 搜索距离顶端、无可行状态、曲线引起的空间交叉分别另行标记。
- 少于 3 个连续有效采样点的片段标记 `insufficient_continuous_support`，避免将孤立强边缘称为连续成果。
- 屏蔽点返回局部基线估计供诊断；无候选返回 CSV 空值/JSON null。它们不与正常段共同平滑，也不跨缺口连接边界矢量。未另做全路宽度平滑或补齐。

## 输出契约

所有 `*_x/y` 坐标和距离均使用 manifest 指定的 UTM CRS/米；left/right 相对于每条线的存储方向，不是地理东西方向。

| 文件 | 内容 |
|---|---|
| `manifest.json` | 输入绝对路径、SHA-256、CRS、影像变换、参数、选择规则 |
| `input_roads.gpkg` | 选中的合并后中心线，保留 road_id |
| `samples.csv/json` | 每点里程、中心/法线、两方法的左右边界坐标、left_distance/right_distance/width、左右及整体 confidence、response、flags、accepted |
| `width_profiles.csv` | 精简的逐道路连续宽度数据，必须同时读取 accepted/flags |
| `measurements.gpkg` | samples 点、两方法 cross_sections、分段 accepted_boundaries 左右线图层 |
| `roads/*_profiles.npz` | 原始 RGB/Lab 横断面、四项特征、综合响应全栅格、两方法距离和置信度 |
| `roads/*.png/csv` | 全部 120 条道路的原图叠加、基线对比、width profile、逐点数据 |
| `cases/*` | 重点案例的局部放大叠加和展开横断面/width profile |
| `summary.json`、`road_summary.csv` | 汇总及每条道路连续性指标 |
| `verification.json` | 导出几何、宽度恒等式、断链和输入哈希检查 |

字段前缀 `optimized_` 表示本方法结果，`baseline_` 表示基线；不把低置信度的诊断宽度当作有效测量。GeoPackage 包含全部诊断横断面，正常左右边界图层仅含通过筛选的连续段。

## 已知适用限制

只能在**中心线位于真实道路内部且两边都在搜索范围内**时用正的左右距离恢复宽度；中心线跑到道路外面时，此参数化本身不成立。没有道路语义和铺装面定义，树冠、建筑、沟渠、人行道、中央分隔带都可能形成更强且连续的边缘。固定颜色与纹理尺度目前只验证了一个约 0.5 m 分辨率影像，不应直接推广到不同传感器或分辨率。

目前没有逐点边界真值，也没有同一评价集上的 SAM-MoLRA 精度比较。因此能判断连续优化是否降低跳变、哪些影像场景明显失败，不能报告真实测宽 MAE、成功率或宣称可替代性已得到精度验证。

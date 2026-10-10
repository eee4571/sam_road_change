# 基于影像证据的道路图修正实验（第一版）

## 本轮：以实际路面和连接关系验收

报告首先给出逐 ROI 视觉结论：中心线是否落在实际路面中央、同一路段的单/双轴表示是否一致、路口是否真连通、是否残留孤立伪线。端点、分量、小环等数值只列为结构诊断，不能据此宣布修复成功。

- `structural` 先做长窄同铺装重复环的整段 canonical axis，再执行既有短链、域重建和延续追踪。所有真实入口固定，环内存在独立地物/分隔带时保持原图。
- 中央轴候选先由骨架位置和方向生成；不再要求错误原轴在原缓存宽度下先通过 RGB 检验。固定多孔径、有限 corridor 搜索结合纵向重复结构和两侧路缘；植被正证据禁止把树带当主轴。没有重新测宽。
- 同一路面内的宽、窄轴方案竞争，抑制车道标线产生的伪轴。重叠连续轴联合拟合，接回旧轴采用渐变；最终实际几何再次评分。
- 原线先按实际接点构图，保留中段支路；事务两侧使用相同的完整 ROI 上下文，避免要素拆分造成虚假的外部断连。
- `--algorithm medial` 可独立检查中央轴阶段。`test_corridor_evidence.py` 包含缓存宽度失配、树带、真实平行路、成对路缘、长重复环和中段支路反例。

运行目录保存实际导入的项目依赖指纹与源码副本，用于区别并发代码变化和实验本身的变化。`visual_review.json` 是逐图检查记录，必须明确未解决处；不是人工标注真值。

```powershell
& road_change_plugin/runtime/env/samroad_env/python.exe `
  road_change_plugin/experiments/image_guided_graph_refinement/visual_comparison.py `
  <上一轮运行目录> <本轮运行目录>
```

该命令生成同范围、同线宽的 `rgb_previous_current.png`，并根据目视记录中的 `cases` 导出局部对比。只生成诊断图，不修改中心线。

## 短弯链、复杂域与主动追踪

`run_multi_roi.py --algorithm structural` 沿用既有中心线骨架、medial axis、支路重挂、RGB route 和事务保护。
新增 `structural_refinement.py`，不调用整区联合平滑：

- 20～60m degree-2 chain 在最多 4.5m corridor 内重拟合；固定接点和 3m 接近段。
- 50m 直径内多类异常组成 repair domain；keep 与收缩/重接/连接/裁剪/拟合的组合方案整体竞争。
- 可信端点主动沿 RGB ribbon 作方向状态束搜索，无须预先存在另一端点；最大前向 60m、横向 18m。
- RGB 连续性、同铺装、中心性、曲率、真实平行道路与外部连通保护共同决定是否提交，最多两轮。

复用已自动选择的十个场景，以免更换 ROI 掩盖退化：

```powershell
& road_change_plugin/runtime/env/samroad_env/python.exe `
  road_change_plugin/experiments/image_guided_graph_refinement/run_multi_roi.py `
  --project project/plugtest --period 20230416 --area 1 --algorithm structural `
  --selection <上一轮 selection.json> --inputs <上一轮 inputs.json>
```

`--inputs` 显式绑定保存的期次记录，并检查全部实际影像、矢量和缓存的大小/mtime 指纹。
用于当前项目结果索引已更新、但需维持可比实验输入的情况；实际资产变动会拒绝运行，不自动混用新旧结果。
省略 `--selection` 时仍按现有几何与场景代理自动选择 6～10 个 ROI。

`assemble_multi_roi.py <运行目录>` 将不重叠 ROI 拼成整期 `final_centerlines.gpkg` 和 `whole_period_before_after.png`。
ROI 内使用毫米节点精度并固定回原始边界接触（3mm 最大吸附容差），修正独立裁切产生的微米接缝；窗外几何不改。
整期连通、几何有效性和窗外变化检查未通过时拒绝导出。实际改动仅限所列 ROI，不代表对整期每条路完成修复。
`revalidate_multi_roi.py` 只用于已保存候选的保护检查复核，不能代替候选重算或目视验收。
相关反例与拼接回归测试在 `test_structural_refinement.py`。

## 街区整体规则化

`run_multi_roi.py --algorithm block` 使用 `block_graph.py` 的街区联合求解；默认 `local` 保留上一轮对照。
共享路口是同一个变量，RGB 横向位置、连续道路方向、曲率和相邻道路间距一起进入稀疏系统。
固定 3/5/8m 与缓存尺度组成影像描述孔径，不将正式测宽作为唯一搜索尺度，不输出新测宽产品。
闭合区域先检查内部是否为同一铺装，再尝试联合收缩和保留外部道路；真实围合道路不能凭小周长删除。
长道路段估计断路进出方向；新增段还须具有相符路面色度和连续两侧结构。
街区模式允许两侧地物具有不同反差方向（例如亮建筑/灰路/暗树荫），仍要求内部均匀且两侧分别有边界；
`local` 对照保留原同向反差描述。`street_route.py` 在方向状态网格上搜索允许转弯的道路走廊，
使用两端已有铺装颜色抑制穿屋顶捷径。没有路缘的共同铺装前庭单独建模，要求短距离、连续匹配材料、低颜色离散和无穿越。

联合拟合出现重复线退化时，将冲突道路固定后重新求解，最多 4 次，不放宽事务检查。
窗外道路与原始端点固定；真实转弯和环路保留。`block_proposal.gpkg` 是最后一次候选，
可能已被撤回，**只有 `final_centerlines.gpkg` 是实际接受的结果**。主方向拟合不是道路真值或自动制图完成证明。
运行目录保存实验 Python 源码快照。输入清单的并发元数据变化与真正的影像/矢量/缓存变化分别核验，后者会中止运行。
相关测试：`test_block_graph.py`。

## 多 ROI 实验

`run_multi_roi.py` 是本轮独立实验入口，自动选择 6～10 个 400～800m 窗口，最后两区为相对正常对照。
几何异常排序结合空间分散约束；粗 RGB 仅用于影像覆盖和场景代理筛选，建筑密集代理不等于建筑分类。
每区从同一正式中心线独立开始；对照也运行全部优化，没有跳过或另设阈值。正式接口和 pipeline 不变。

```powershell
& road_change_plugin/runtime/env/samroad_env/python.exe `
  road_change_plugin/experiments/image_guided_graph_refinement/run_multi_roi.py `
  --project project/plugtest --period 20230416 --area 1 --count 10 --roi-size 600
```

`--selection-only` 只保存选择清单、分布图和原始诊断图。
`--selection <已有 selection.json> --only roi_02 roi_04` 复用绑定输入指纹/CRS 的窗口进行局部迭代；不复用上一区的修改结果。
新建 `outputs/multi_<UTC>_<id>/`，逐区包含 `before_after.png`、`operation_*.png`、修复前后 GPKG（context / roi 两层）、
`changes.gpkg` 的 added/deleted/replaced_old/replaced_new 层、审计、拓扑统计和收敛信息。
总目录生成 `MULTI_ROI_REPORT.md`、`summary.csv`、`evidence_distributions.json`；可在 `visual_review.json` 填写逐区目视结论后重新生成报告。

新模块：`roi_mining.py` 选择窗口；`color_ribbon.py` 统一颜色断面/中心性；`local_graph_optimizer.py` 生成并比较局部图方案；
`graph_transaction.py` 逐操作保护；`multi_roi_report.py` 生成报告与细节图。复用已有中央轴重建、图原语、影像读取和方向约束路由。

局部假设含 keep、中央轴替换/支路重挂、细窄回环联合 merge/rewire、固定接点的多孔径 refit、端点对及端点至道路中段 connect，
以及需要肯定植被证据的终端伪支路 delete。低 SAM、短长度或小周长都不能单独触发删除。RGB 高分也不能单独通过：
检查窗外几何、固定接点连通、新悬空端点、小环、重复线、跨层和真实平行道路；发现退化便撤回该事务。
保留原有缓存宽度字段；离散影像描述孔径不是重新测宽结果。每区最多 3 轮，无新可接受修改时终止。

试验不是道路语义识别器：河渠、屋顶、绿化、阴影可形成高分 ribbon。目视失败案例与正常对照稳定性必须连同分数一起评价，不能将拓扑计数下降称为准确率提高。
相关回归测试：`python -m unittest discover -s road_change_plugin/experiments/image_guided_graph_refinement -p 'test_*.py' -v`。

## 实际修复实验

新增 `run_repair.py`，与下文第一轮诊断 CLI 并存；不改变正式 pipeline、正式中心线或缓存。
它实际重建中央轴、重挂外部支路、处理窄小伪环、按方向约束搜索 RGB gap route，并对固定拓扑节点之间的折线做小范围 RGB 一致性平滑。
支持 `--bbox`、`--center X Y --radius R`，不指定时沿用候选密集 ROI 选择。坐标均为加载输入后选定的米制 CRS。

```powershell
& road_change_plugin/runtime/env/samroad_env/python.exe `
  road_change_plugin/experiments/image_guided_graph_refinement/run_repair.py `
  --project project/plugtest --period 20230416 `
  --bbox 258791.0108 2613777.3135 259591.0108 2614577.3135 --whole-period
```

`--whole-period` 先修复指定 ROI，再以默认 900m 分块扫描其余区域（`--tile-size 400..1500`）；不指定则只修改 ROI。
无论哪种模式，`final_centerlines.gpkg` 都导出完整期次网络，实际扫描范围以 `inputs.json` 和各 tile 的 ROI 为准。
每次创建独立 `outputs/repair_<UTC>_<id>/`，包括：

- `final_centerlines.gpkg`、`baseline_centerlines.gpkg`；
- `changes.gpkg`：added、deleted、replaced_old、replaced_new（无对应修改时写空层）；
- `whole_period_before_after.png`、各 `tile_*/before_after.png` 和路口 `local_*.png`；
- `audit.json`、`summary.json`、输入指纹、源码 SHA256、候选、分块验收和收敛记录。

输出沿用输入的缓存宽度属性，没有在新轴上重新测宽；这些属性不能当作修复后道路的重新测量结果。

算法先从同层侧轴对和已有轴线链提供位置/方向先验，在原始 RGB 上寻找横向均匀、有两侧反差且纵向连续的道路带。
允许在同方向、已有网络附近继续召回跨路口的道路段；单靠低 SAM 不否定道路。SAM/MOLRA 只加正分，正式 surface 和缓存宽度仅作辅助/搜索尺度。
比较同一搜索尺度下旧轴和新轴的 raw RGB support，并为同路带侧轴替换保存横断面连续性证据；有稳定非道路隔离带的真实平行道路保留。
重建路口采用同层 medial axis 的交点和路口范围，所有外部支路接触点保留并重挂；没有按示意图未标道路来删除厂区内部道路。
几何无效、自交增加、连通分量增加、重复候选长度增加的分块撤回，候选及失败原因仍保留。`applied=false` 的审计记录不是最终修改。

限制：RGB ribbon 是结构证据，不是新的道路语义模型；明亮屋顶/院落、阴影、未标注层级仍可能造成歧义。
小环/端点/近平行候选数不等于错误数，拓扑改善和 ribbon 分数提高也不等于精度或召回率。完整成果、图和报告应一起查看。
第一轮诊断仍可由 `run_experiment.py` 原样复现；新修复实验不使用旧的“低 SAM 即负证据”删除策略。

```powershell
& road_change_plugin/runtime/env/samroad_env/python.exe -m unittest discover `
  -s road_change_plugin/experiments/image_guided_graph_refinement -p 'test*.py'
```

独立、离线、只读项目输入的实验。**没有接入正式 pipeline，也不发布或覆盖正式成果。**
不调用 SAMRoad、MOLRA、IR-MAD、测宽或道路面重建任务。新增计算限于候选生成、已有影像的局部描述、路径搜索和诊断统计。

## 运行

在仓库根目录使用现有插件 Python 环境，无需安装模型或新建环境：

```powershell
$python = 'road_change_plugin/runtime/env/samroad_env/python.exe'
$experiment = 'road_change_plugin/experiments/image_guided_graph_refinement/run_experiment.py'

# 检查期次、缓存和坐标系，不运行实验
& $python $experiment --project project/plugtest --period 20230416 --inspect

# 默认自动选择候选较密集的 800m 正方形，不跑整期修正
& $python $experiment --project project/plugtest --period 20230416

# 首轮目视选出的厂房密集 ROI：800m × 800m
& $python $experiment --project project/plugtest --period 20230416 `
  --center 259191.0108 2614177.3135 --radius 400

# 同一区域的 bbox 写法
& $python $experiment --project project/plugtest --period 20230416 `
  --bbox 258791.0108 2613777.3135 259591.0108 2614577.3135
```

`--bbox`、`--center` 坐标使用 `--inspect` 输出的**米制分析 CRS**，不是经纬度，也不一定是原始影像 CRS。
`plugtest/20230416` 的分析 CRS 是 **EPSG:32650**，原始影像是 EPSG:4548，不能混用。
`--radius` 是正方形半边长，不是圆形裁剪半径；500～1000m ROI 对应半径 250～500m。
原始中心线为经纬度或非米单位时选择当地 UTM。多区域有同一期次时必须提供 `--area <grid>`，不会任取一个。
可用 `--config <json>` 指定完整配置。参数、输入路径、大小、mtime、ROI、坐标系及源码 SHA256 均写入运行记录。

自动选区通过本期矢量候选密度选择，不识别工业用地；仅选择阶段读取本期中心线，栅格分析和修改仍限定在 ROI。
默认增加 160m 证据上下文，保留相交原始要素的完整几何以判断真实度数；修改必须完全处于 ROI 内。
栅格以 0.75m 采样到内存，超过 2000 万分析像元会拒绝运行。没有全期模型运行入口。

## 输入发现与复用

从 `<project>/_work/current/pipeline_result.json` 的当前 `period_results` 解析指定区域/期次。
不会遍历所有项目文件，也不会悄悄选旧的 run 或其他期次；缺少当前正式中心线直接报错。

| 输入 | 用途 |
|---|---|
| `centerlines`、`surfaces` | 原样 baseline、图节点/边、已有宽度和表面支持 |
| `products/road_probability.tif` | 首选已有 SAMRoad probability；缺失时匹配本 run 的 tile probability PNG 和参考影像 |
| `width_review/*_summary.json` | 通过现有 `molra_sources` 获取 MOLRA mask；缺失记录，不生成 |
| `source` 的影像或影像 TXT | 只读原始影像；窗口重投影、灰度归一化和梯度均为内存描述，不重新执行 IR-MAD |
| `products/raw_width/width_observations.gpkg:point_profiles` | 沿线附近且方向匹配的**已有**宽度、置信度、来源和质量标记 |
| `regional_products.gpkg`、`width_profile_cache.json` | 可用性和文件指纹登记；本版正式轴线已有 width_m，直接使用点观测，不重复从区域中间成果构建宽度 profile |

每条候选记录缺测覆盖率。缺失值写 JSON `null`，不作为“明确无道路”证据。
正式道路面由正式轴线派生，存在自证偏差；它参加路径 cost 和审计，**不能单独授权删除、连接或合并**。
本轮 MOLRA 缓存不可用，实际使用原始影像、SAMRoad probability 和已保存的测宽观测。
没有生成实验道路面；因此也没有触发任何重新测宽或正式道路面重建。

## 实验设计

每次执行完整生成以下六组，使用相同 ROI、相同冻结证据与相同几何候选定义：

| 组 | 从 A 独立起步 | 行为 |
|---|---|---|
| A / `baseline` | 是 | 当前正式中心线，输出仅作坐标转换和 ROI 裁剪；不修正 |
| B / `geometry_only` | 是 | 同一套几何候选，不用影像 veto；顺序 spur → gap → duplicate |
| `single_spur` | 是 | 只做影像支杈判别 |
| `single_gap` | 是 | 只做影像断路连接 |
| `single_duplicate` | 是 | 只做重复线/平行道路判别 |
| C / `image_guided` | 是 | 三项顺序组合，最多 3 轮；后续只重新考虑改动邻域候选 |

B 是**共享候选的几何消融**，不是重新运行正式算法：复用正式图构建和重复线候选实现，支杈使用相同的 `min(30m, 3×width)` 几何尺度，断点采用方向/距离/横向偏移/宽度筛选。
B/C 均保留交叉接触、同端点竞争、分叉连接保护；这样差异来自影像判定和影像路径，而不是不同的安全规则。
正式成果本来已经经过几何处理，因此这些对照测的是“对当前成果继续修正”的增量效果。

### Experiment 1：支杈

- 真正的 degree-1 端点沿 degree-2 边追踪至分叉，按整条支杈长度筛选；不是把每段短线都当支杈。
- 每 2.5m 采样，优先评价端部 65%，避免主路根部的支持掩盖假支杈。
- 复用灰度和梯度，沿缓存宽度的横断面提取左右方向性边缘；不搜索或重算新道路宽度。
- 记录 bilateral edge score、SAM probability、MOLRA、正式 surface、连续支持比例、最大无支持长度，以及缓存宽度 CV/质量/观测行。
- 有效影像覆盖不足 → review；独立支持 ≥85% 且双侧边缘 ≥0.45 → keep；缓存宽度质量 ≥0.16 也保护道路。
- 只有独立支持 ≤15%、双侧边缘 <0.2、且没有较强缓存宽度支持时删除；其余 review。

### Experiment 2：断路

- 两个 degree-1 端点，距离 1～100m、两端方向 cosine ≥0.94、横向偏移 ≤12m、宽度比 ≤2；保留层级区分。
- 复用 `_evidence_route`，在方向定义的 corridor 内搜索。cost 使用 SAM/MOLRA、由缓存宽度探测的 RGB 双侧边缘和正式 surface。
- 路径经过简化后重新评分，要求持续影像支持、无支持段 ≤7.5m、方向 ≥0.94、长度/直线距离 ≤1.2、最大局部转角 ≤50°。
- **没有“短距离默认连接”**；低概率、无影像、曲率/方向不足都保持断开。
- 新路径不能交叉或接触其他道路，也不能与已接受连接竞争同一端点。暂不做 junction rewire。

### Experiment 3：重复线与平行道路

- 对 `_collapse_duplicates` 做 dry run，使用其近平行/重叠候选，忽略它生成的修改结果。
- 沿两条轴线建立配对横断面，保存 SAM、MOLRA、正式 surface、原始灰度 profile。
- 两轴间有持续道路支持且没有稳定原始影像间隔带 → 单 ribbon 候选；两轴均支持、中间持续低支持且宽 ≥1.5m → 保留两条。
- RGB 稳定间隔带也禁止合并，即使 probability 模糊地填满了间隔。该 cue 不是语义分类器，阴影也可能触发保守保留。
- 缺测、亚像元间距、歧义 → review。部分重叠或删除会悬空分叉接点 → review。
- 第一版“合并”只删除完整、安全的冗余轴线，保留另一条已观测轴线；不平均到两路之间，不自动迁移分叉。

### Experiment 4：组合

原图只构建一次 noded graph。删除仅去边；连接被要求没有中间接触后才加边，保留现有宽度，不重新 noding/测宽。
每阶段更新邻接关系；下一轮候选仅在上一轮修改的 160m 邻域内检查，最多 3 轮或没有新修改停止。
小环和复杂 junction 只统计坐标/几何，当前不处理。小环指标是 ≤60m 周长的几何候选数，不等于已确认的错误数。

### 补充诊断：道路两侧伪轴线、中央主轴漏提

用户指出工业 ROI 东北侧的西北—东南道路存在这个问题。两条边侧线间距约 30～35m，超过正式 duplicate 候选的最大 12m 搜索距离；只删除其中一条也无法得到正确主轴。

`ribbon_diagnostic.py` 在同一 ROI 扩大到 45m 搜索方向一致的侧线对，检查配对横断面中间是否为连续、较均匀且与两侧有反差的 RGB ribbon。
该诊断**不要求 SAM 高概率**，以免再次漏掉“原始影像明确、模型漏检”的道路。
在支持段内利用原始灰度相对两侧的对比度重心提出内部轴线，检查中央是否已经有轴线，输出 `boundary_axis_diagnostic/`。
它不估计新宽度、不平均修改原路、不自动把 proposal 加入 C；所有候选 `decision=review`、`auto_applied=false`。
完整替换侧线还需要辨认厂区支路如何接入新主轴，属于下一步 junction rewire；亮度平台也可能来自广场/屋顶，因此不能仅凭该 cue 自动修改。
图中青色为现有轴线，品红色为待复核的 RGB 主轴建议；GPKG 图层名 `review_only`。

## 输出和人工评价

所有运行目录固定在本目录 `outputs/<UTC时间>_<随机ID>/`，没有指向项目成果的输出参数。输出 symlink 若越出实验目录会拒绝。

```text
inputs.json                 # 缓存、缺失项、ROI/CRS、配置、输入与代码指纹
baseline/                   # centerlines.gpkg / metrics.json / review.png
geometry_only/              # 另含 audit.json / edits.json
single_spur/                # 另含 candidate_reviews/ 每个候选局部图与 profile
single_gap/
single_duplicate/
image_guided/               # 组合结果，accepted_edits/ 展示每个自动修改
boundary_axis_diagnostic/   # RGB 中央漏轴诊断，review.png / audit.json / axis_hypotheses.gpkg
comparison/comparison.md    # A/B/C 数值及 C-B
audit.json                  # 各组完整候选审计、决策原因与未开展的实验
summary.json                # 各组统计、耗时、输入指纹不变检查
```

GPKG 为米制 CRS，JSON 的 WKT/几何坐标也使用 `inputs.json.metric_crs`；`edits.json` 明确附 CRS，不冒充经纬度 GeoJSON。
baseline 导出正式要素属性；修正结果保留节点化来源索引、宽度和层级，`local_source_index_map` 可追溯原始 feature。
候选含 ID、类型、几何分数、影像分数/逐站 profile、决定、原因、原/结果几何；duplicate 另存对照轴线。
全图颜色：黄色 baseline、青色结果、红色删除、绿色新增、紫色 review。候选图同时绘制原几何和决策几何。

评价关注：component、真实 degree-1 端点、spur 数/长、duplicate 数/长、小环、连接数、删除长度、改动位移、长度加权影像分数。
component 在 ROI 裁剪图中计数；端点按原图度数统计，排除裁剪制造的断点。新增线位移相对于 A，合并位移相对于保留轴线；纯删除无对应位移，输出 null。
**这些是无真值诊断指标，不是 precision/recall。** 全图平均影像分数会受删除长度和选择偏差影响；应同时看删除长度、计分覆盖率、候选图和单项结果。
人工应逐条标注 `true/false/uncertain`，尤其检查真实厂区出入口、小路和真实双幅道路是否被误删，再比较 B/C 决策。
没有人工标注前，不能把端点减少或影像分数提高称为准确率提高。

## 文件职责与验证

- `data.py`：当前结果清单和只读缓存定位；`bootstrap.py`：导入插件 engine。
- `graph.py`：共享图与几何候选；`evidence.py`：窗口影像/缓存证据。
- `ribbon_diagnostic.py`：宽路双侧伪轴与中央漏提的非破坏性补充诊断。
- `refine.py`：实验决策与局部改动；`report.py`：评价、JSON 和图；`run_experiment.py`：CLI 与各对照组。
- `config.json`：首版尺度/阈值；`test_experiment.py`：合成回归。实验代码不需要导入正式 pipeline。

```powershell
& $python -m compileall -q road_change_plugin/experiments/image_guided_graph_refinement
& $python -m unittest discover -s road_change_plugin/experiments/image_guided_graph_refinement -p test_experiment.py -v
```

测试使用小型内存几何/临时缓存栅格，覆盖支杈、连接、真假平行道路、边界/层级/分叉、缺测、严格 JSON 和原始文件字节不变。
测试临时数据也只写 `outputs/`。`.gitignore` 排除 outputs、影像/矢量、缓存和日志；源码、README、配置与测试可跟踪。

首轮实际结果见 [FIRST_RUN.md](FIRST_RUN.md)。仍待验证：更多期次、不同传感器、可用 MOLRA 的项目、实际正例连接/合并的人工精度、复杂接点与部分重复线。
# 整条道路走廊重建（2026-10-10）

**道路表达修正：连续规则不等于单轴化。** 宽路具有反复出现的双线先验时，沿整条道路联合拟合两条有独立身份的中心线；分向标线也可形成双向道路表达，不要求必须有绿化隔离带。双线一起提交或一起撤回，不能逐段切换单双线。只有宽路面证据、无法确定线数时，保留原网，不再默认合成一条轴。窄路继续保留单线表达。最新结果与限制见 [CARRIAGEWAY_CONTINUITY_REPORT.md](CARRIAGEWAY_CONTINUITY_REPORT.md)。下方单轴阶段的历史报告不能作为该双线道路的验收结果。

最新主路实验入口为 `run_corridor_experiment.py`。它复用已有骨架和 RGB，在同一道路的纵向范围内共同估计轴线，整体替换旧碎片、重新挂接支路；不运行模型或重新测宽，不接入正式 pipeline。

```powershell
& road_change_plugin/runtime/env/samroad_env/python.exe `
  road_change_plugin/experiments/image_guided_graph_refinement/run_corridor_experiment.py `
  --project project/plugtest --period 20230416 --area 1 `
  --inputs road_change_plugin/experiments/image_guided_graph_refinement/outputs/multi_20260929T115744Z_55728afa/inputs.json `
  --bbox 260060 2614180 260500 2614620
```

`--inputs` 只用于恢复对应项目/期次的缓存发现记录；每次运行重新记录**当前**缓存指纹，不冒充旧轮数据未变化。项目、期次、区域必须一致。bbox 使用加载中心线的米制 CRS（本项目 EPSG:32650）。不传该参数时直接按当前项目清单发现缓存。

多 ROI 可用 `--roi-list <json>` 替代 `--bbox`。JSON 结构为 `{"regions": [{"id": "roi_01", "category": "wide_road", "bbox_m": [xmin, ymin, xmax, ymax]}]}`。每区独立从当前正式成果出发；只允许不重叠的 ROI 合成整期成果。批量导出的整期文件仅修改列出的 ROI，不能据此宣称全期均已修复。

- `whole_corridor.py`：完整走廊候选、道路表达判定、共用平顺曲线的双线拟合、整段 RGB 对齐、真实 T 接入端部约束；候选孔径是影像描述尺度，不输出为新测宽成果。
- `medial_repair.py`：复用已有道路走廊替换、支路重挂和真平行道路保护；完整走廊不再被两个内部接入点截短。
- `rgb_route.py`：抽取共享的成对路缘描述，不改变原有 longitudinal ribbon 调用行为。
- `run_corridor_experiment.py`：同尺度旧/新轴 RGB 比较、相交走廊联合事务、失败回退、独立输出与源码快照。
- `graph.py`：保留明确的 carriageway 身份，两条不同路幅不算同路幅重复；同一身份内部的重复仍检测。双线表达并不免除断连、交叉、范围损失等保护。
- `test_whole_corridor.py`：暗路面、碎片归并、内部支路保留、远处屋顶候选拒绝、真实平行道路保护。

每区输出 `before_after.png`、`baseline_centerlines.gpkg`、`final_centerlines.gpkg`（`context` 和 `roi` 图层）、`changes.gpkg`（added/deleted/replaced_old/replaced_new）、`audit.json`、候选及检验记录。批量运行另输出整期 `final_centerlines.gpkg`（`centerlines` 图层）和全区叠加图。

该实现主要针对具有稳定方向先验的道路走廊，尚不能把完整缺失的复杂弯路、遮挡下的厂区通行关系自动视为已解决。是否改善须查看原始 RGB 叠加图，拓扑计数仅用于防退化。

## 内部道路平直化（2026-10-10）

`run_internal_experiment.py` 读取已经生成的实验 ROI，不从正式成果重新覆盖上一轮双线主路。`internal_axis.py` 复用现有 RGB 描述、局部回环重建、degree-2 chain 和事务检查：先处理环内仍为同一铺装路面的两入口细长伪环，再比较固定接点的直路段拟合。真实转角附近保留原几何；每个改变的直路段单独检查 RGB，避免整链平均分掩盖局部偏离。带 `carriageway_id` 的既有主轴锁定。

```powershell
# 在实验目录运行；所有结果写入新的 outputs/internal_*/
& ../../runtime/env/samroad_env/python.exe run_internal_experiment.py `
  --source-roi outputs/corridor_multi_20261010T054258Z_b4ca41fc/main_industrial `
  --full-period-source outputs/corridor_multi_20261010T054258Z_b4ca41fc/final_centerlines.gpkg
```

`--source-roi` 目录必须有 `inputs.json` 和 `final_centerlines.gpkg` 的 `context` 图层。可选的 `--full-period-source` 必须与 ROI 的修复前几何匹配，整期输出位于新运行的 `full_period/final_centerlines.gpkg`，只替换该 ROI。局部矢量、变化、审计和前后图仍在新运行根目录。

这里解决的是有现成骨架的内部直路抖动及同路面细长伪环；不会凭直线外推补造缺失道路，也不代表完整路网已通过视觉验收。实际结果见 [INTERNAL_AXIS_REPORT.md](INTERNAL_AXIS_REPORT.md)。

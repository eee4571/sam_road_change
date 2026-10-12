# RGB 测宽内存安全与异常隔离验证

日期：2026-10-12。基于 `main` 的 `a954bd9`；实际插件路径为 `fast + raw_image`。同时提交的局部异常边界改造见 [WIDTH_EXCEPTION_BOUNDARIES.md](WIDTH_EXCEPTION_BOUNDARIES.md)。

## 根因和改动

在 `raw_boundary_core.ImageReader.features()` → `raw_feature_cache.FeatureCache.features()` 中，旧实现按**整条道路全部断面的外包矩形**分配 RGB、Lab、灰度、边缘、纹理和掩膜数组。稀疏跨切片拼接的影像网格、长斜线或高分辨率影像可使这个矩形远大于实际需要读取的道路带；限制 LRU 缓存不能限制构建这些数组的瞬时内存。这是代码中已确认的无界分配路径，与所报 40.5 GiB 分配失败相符，但没有故障电脑原始 traceback，不能断言它是那次错误的唯一位置。

另在 `raw_width_reconstruction.smooth_side()` 发现 n×可信断面数的密集距离矩阵。改成排序查找最近可信点，内存为 O(n)，保留原来等距取左侧/较早锚点的规则；优化目标、求解器、断面评分公式不变。

| 文件 | 作用 |
| --- | --- |
| `code/engine/width/raw_memory.py` | 新增进程共享工作预算、分配前检查、GDAL 缓存生命周期和所有 RGB 读取器共享的特征缓存上限。 |
| `raw_boundary_core.py` | 沿道路连续断面分窗，先规划后分配。每个断面保持原坐标、搜索距离、采样间距和证据 halo；没有缩放/错位匹配影像。正常小道路保留原完整窗口。 |
| `raw_feature_cache.py` | 在读窗和构造特征前校验尺寸；单独特征调用也需要内存准入；缓存失效后仍被 Worker 持有的数组计入其工作预算。关闭读取器释放缓存。 |
| `raw_image_backend.py` | 准入覆盖单路采样、剖面转换和重建；释放不再需要的完整采样数据。预算拒绝走原有多级回退；拆窗估计标 C、`bounded_rgb_estimate`、记录原因。 |
| `raw_width_reconstruction.py` | 消除最近点查找的平方内存矩阵。 |
| `width_reliability.py`、`fast_pipeline.py` | 内存策略进入期次/区域测宽缓存身份，原生分配失败仍属系统异常。 |

文件名未写目录时位于 `code/engine/width/`。

## 预算与边界

- 单个影像特征窗口最多 **524,288 像素**，单边最多 **4,096 像素**；点特征块不超过 512×512。
- 进程共享算法预算 **512 MiB**：并发工作区 352 MiB + 特征缓存 128 MiB + GDAL 块缓存 32 MiB。工作区估算包括特征临时数组、被缓存逐出的在用数组、断面输出、候选状态、求解器/表格副本和密集轴线插值。
- 多个 Worker 在分配前按需占用同一个工作预算；容量不足等待。单条道路的需求超过总工作容量时，直接产生 `RoadMemoryLimit`，没有尝试巨额分配。超过断面数量/密集插值上限，或单个断面就无法容纳，也转入 C 级回退。
- GDAL 默认缓存可能按主机内存比例分配（本机原设置约 1.6 GiB），因此必须单独纳入预算。最后一个 RGB 读取器关闭时恢复原设置，含打开文件失败路径。
- 拆窗后仍按整条道路做原来的连续宽度优化。Canny 连通性依赖窗口，不能宣称拆窗后与整幅大裁剪完全等价，因此拆窗道路保守标为 C，进入人工复核，不参与自动拓宽/变窄结论。
- **512 MiB 不是整个 Python 进程 RSS 的硬上限。** Python/地理库运行时、原生分配器保留页、输入路网、累计输出记录及后续正式成果合并不在此工作预算内。使用额外 RSS 测试检查实际峰值，避免只验证预算计数器。
- 实际 `MemoryError`、`Unable to allocate`、OpenCV 原生内存错误、I/O/权限/依赖/CRS 网格一致性失败和取消仍中止；不会把它们伪装成正常道路错误。已提交检查点保留。

## 执行结果

本次 **179 个测试全部通过**：

| 测试文件 | 数量 | 结果 |
| --- | ---: | --- |
| `test_rgb_memory_safety.py` | 11 | 通过 |
| `test_width_local_exceptions.py` | 15 | 通过，含 48 组局部异常注入和真实子进程退出码 |
| `test_width_production.py` | 33 | 通过 |
| `check_width_resilience.py` | 17 | 通过 |
| `check_formal_period_products.py` | 17 | 通过 |
| `check_fast2_structure.py` | 23 | 通过 |
| `check_processing_optimizations.py` | 25 | 通过 |
| `test_width_warning_ui.py` | 3 | 通过 |
| `test_ui_processing.py` | 35 | 通过 |

新增内存测试覆盖：

1. 正常道路：与原完整裁剪、原平方矩阵最近点实现逐值对照，宽度/左右距离/来源/质量及几何完全一致。
2. 超长道路和稀疏大网格：两个 128×128 TIFF 构成约 120000×120000 的网格外包范围，旧 RGB float32 数组单项理论需求约 160.9 GiB；新实现不进入采样即回退，合法中心线完整保留。恢复运行复用两个已完成道路检查点。
3. 超出单路工作预算（仍未超过 20000 断面）的长道路：在数组构造前回退为配置默认宽度和 C。
4. 1 cm 分辨率异常单断面：3124×3124 窗口在构建特征前拒绝，保留道路，配置默认宽度 7.5 m。
5. 跨两块 TIFF 的道路强制拆窗：每窗 ≤4096 像素，宽度与平直道路完整窗口对照在 1e-10 容差内一致，几何/总长度/切片关联一致，结果标 C。
6. 100000 个站点的最近锚点查找：完成线性内存处理；小数据全矩阵对照以及等距行为一致。
7. 多读取器共享特征缓存、GDAL 设置恢复、内存策略变更使测宽缓存失效。
8. 四线程独立进程压力测试：实际执行 RGB 特征计算/采样/重建，非 Mock。四条大外包矩形斜线均安全完成 C 级结果。

最终压力测试实测（Windows）：

| 指标 | 数值 |
| --- | ---: |
| Worker 请求数 / 最大同时准入数 | 4 / 2 |
| 工作预算峰值 / 上限 | 329859840 / 369098752 bytes |
| 完成后的在用工作预算 | 0 |
| 最大实际特征窗口 | 153664 pixels |
| 初始化后的进程峰值 RSS | 460525568 bytes（约 439 MiB） |
| 完成时进程峰值 RSS | 662274048 bytes（约 632 MiB） |
| 峰值 RSS 增量 | 201748480 bytes（约 192 MiB） |

压力测试另断言 RSS 增量 <640 MiB；该阈值含运行时/原生分配器余量，不等于预算本身。测试不会尝试真实分配 40 GiB。

## 已有项目回归

`plugtest`，区域 1、期次 20221020，600 米子集：44 条输入中心线 → 55 条恢复线；4 条测宽链、141 个断面。最终代码 RGB 测宽约 1.44 秒（单次检查，不作性能基准）。输出：`project/plugtest/_work/_processing_check/period-20221020-54807a8c/`。

与内存改造前缓存检查结果 `period-20221020-8cb80586` 对照：141 行的 `final_width`、`final_left_distance`、`final_right_distance`、`final_confidence`、`width_source`、`quality_grade` **逐值完全相等**，几何相同。

没有执行道路提取模型，没有替换正式项目成果。该检查证明此缓存子集正常，不代表已在故障电脑复现全部数据。

## 运行与续跑

更新主分支代码后，使用原 GUI 的续跑/测宽重跑即可，无新增 CLI 参数或成果格式。新实现身份只使相关测宽、区域产品和导出缓存重新验证，不删除道路提取、概率图、掩膜或模型推理成果。旧算法道路检查点不能冒充新算法结果；新版本中已原子完成的正常/C 级道路检查点均可复用。

查看 `raw_width/summary.json` 的 `memory` 字段和原有 `exceptions`/`roads` 明细；每条道路检查点仍在 `raw_width/chains/<identity>/`。系统失败查看 `failure.json`。GUI 继续显示警告完成、异常数量及详情入口。

```powershell
runtime/env/samroad_env/python.exe tests/test_rgb_memory_safety.py -v
runtime/env/samroad_env/python.exe tests/test_width_local_exceptions.py -v
runtime/env/samroad_env/python.exe tests/check_existing_period.py ../project/plugtest
```

## 未验证和剩余风险

未运行完整耗时模型、完整多期任务或故障电脑的数据。异常几何导致的 GEOS 原生崩溃、系统真正资源耗尽、不可写磁盘、损坏依赖/网格和未能证明局部性的共享状态异常仍可中止，这是保留的正确失败边界。更大区域输入/累计成果及正式路网合并的总内存仍随数据规模增长；本次预算覆盖的是 RGB 测宽的并发工作区和缓存。

若需确认原始 40.5 GiB 错误的精确行号和远端环境，还需该电脑完整 traceback、版本、影像尺寸/CRS/分辨率以及失败道路几何或相应缓存。当前修复和合成验证不依赖重新运行模型。

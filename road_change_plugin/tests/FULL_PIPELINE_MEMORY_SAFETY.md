# 全流程内存安全改造与自适应并行验证

日期：2026-10-12。对照基线：`9b6dc28a0844a6b0520460ec85267a71e1a73d17`。
机器可读结果：[FULL_PIPELINE_MEMORY_RESULTS.json](FULL_PIPELINE_MEMORY_RESULTS.json)。

## 实施范围

修改的是插件实际的 `Fast + raw_image + IR-MAD` 调用链。沿用已有 RGB 有界窗口、逐道路容错、C 级回退及切片检查点，不替换正常测宽算法。此次覆盖输入规范化、IR-MAD、SAMRoad/TopoNet、区域恢复、正式构面、变化匹配、最终跨期协调、长时序、评价、日志和并行预取。

| 风险 | 修改文件与处理方式 |
| --- | --- |
| 解码前无尺寸检查、超宽 IR-MAD 条带、多影像读取器常驻 | `user_pipeline.py`、`irmad_core.py`、`raster_memory.py`、`fast_patch_verification.py`、`raw_feature_cache.py`：先检查网格及原生压缩块；规范化窗口最多 4096×4096，IR-MAD 窗口最多 524288 像元；读取器/VRT 按需打开，GDAL 缓冲有界。规范化读取失败明确传播，不能记成无数据缓存。 |
| 模型全切片特征和掩膜长期占 GPU、密集拓扑查询按批次最大节点数扩张 | `samroad/inference_memory.py`、`inferencer.py`：特征按字节预算移到 CPU/无损 NPY 临时文件，第二遍逐批取回；TopoNet 先按点数划分 CPU 批次，再按独立查询组分块，保留每组全部邻居。释放上一批 GPU 引用。输入和 GSD 目标网格在解码/放大前检查。 |
| 超长中心线密集采样、走廊拟合大广播和全历史 DP | `geometry_memory.py`、`road_axis_*`、`canonical_road_*`、`auto_change_geometry.py`、`road_track_corridors.py`：分配前估算采样数；普通道路保持原采样。极长道路保留原始顶点，必要时 C 级临时构面。走廊证据广播每次 256 个样本，DP 仅保留当前成本，回溯超过 16 MiB 使用磁盘数组；模型成员和候选排序转到磁盘。 |
| 全期道路记录、多期图层、宽度剖面与变化对同时常驻 | `disk_records.py`、`fast_pipeline.py`、`fast_auto_v2.py`、`fast_gt_reconciliation.py`、`temporal_road_analysis.py`：SQLite 增量记录/索引/排序，2048 条批写；最多缓存两期图层和一个可变变化对；每道路读取跨期记录；几何摘要逐对象计算，保留原缓存摘要格式。 |
| 所有期次空间索引和道路面同时加载 | `vector_evidence.py`、`road_axis_quality.py`、`canonical_road_surface.py`：期次空间索引按需复用；连接证据按局部轴线窗口读取、裁取和合并。 |
| 大型并集及逐线段 GEOS 对象膨胀 | `geometry_recovery.py`、`road_network_products.py`、`auto_presence_candidates.py`：256 个对象一组进行分层并集；覆盖索引按 256 条线段建块，命中后逐线段精确计算，避免每个顶点永久创建一个 GEOS 对象。 |
| 评价全外包矩形及对象候选笛卡尔规模 | `evaluation_memory.py`、`width/road_change_detection.py`、`gt_road_geometry.py`：栅格化之前检查最终行列乘积；匹配连通分组用并查集，贪心候选在磁盘排序，保留原 IoU 与平局顺序。超预算精确骨架指标明确 unavailable/null，不能填 0 或产生虚假精度。 |
| 独立预取各自抢资源、Future 异常被忽略 | `resource_policy.py`、`batch_runtime.py`、`bounded_pipeline.py`、`raw_image_backend.py`：跨进程 SQLite 原子准入、为前台保留同等余量、每次预取重新评估可用内存；低资源串行，资源恢复后重新允许预取；RGB worker 自适应。共享单一模型进程，CUDA batch 参考可用显存和配置预算。预取结果消费前持续保留 RAM 预约，所有异常路径释放预约。 |
| 巨型日志或缺少换行的 stdout | `plugin/runner.py`、`plugin/log_memory.py`、`plugin/widget.py`：分块读取与有界尾部展示；异常超长结构化事件报错，不能静默截断后报告任务成功。 |

`formal_road_products.py` 与 `resources/backend_snapshot.json` 已更新相关代码身份。正式图层字段、CRS、目录及 manifest 契约保持兼容；评价报告增加明确的资源受限原因。长时序额外禁止 C 级观察及历史变化文件中的不可靠宽变重新成为确定结论。

## 资源边界与运行方式

默认直接运行现有插件即可，`ROAD_PIPELINE_PARALLEL=auto`。不需要删除已有模型、概率图、掩膜或道路提取缓存。

可在启动插件的环境中配置：

```powershell
# 可选示例；单位 MiB。配置是调度上限，并不代表机器实际有这些空闲资源。
$env:ROAD_PIPELINE_RAM_MB = '8192'
$env:ROAD_PIPELINE_VRAM_MB = '6144'
$env:ROAD_PIPELINE_PARALLEL = 'auto'
# 串行诊断时改成 'off'；正常使用恢复 'auto'。
```

RAM 默认准入上限为物理内存的 60%，同时检查实时空闲量并保留至少 512 MiB/物理内存 10% 的系统余量。跨期预取预约 3 GiB，并要求前台仍有同等余量；普通预取预约 2 GiB。RGB 仍受已有 512 MiB 算法预算控制。特征 CPU 缓存按可用 RAM 和配置预算取较小值，最多 1 GiB，其余无损落盘。VRAM 预算扣除已保留的 Torch 内存；只允许一个 CUDA 生产者，批次至少为 1，无法容纳模型/单批次时明确失败。

临时特征、记录、排序和 DP 文件使用系统临时目录，退出时释放；磁盘不足和权限错误明确失败。已有完成检查点可用于续跑，但临时文件本身不作为跨任务缓存。局部测宽完成仍按现有质量状态复用；影响正式算法的代码身份变化仅使相关下游成果失效。没有清理或覆盖原项目推理成果。

以下边界是明确的安全拒绝/降级，而非任意缩放影像：

- 单次推理输入或 GSD 目标最多 25 MP；CLI 的 0 不再取消此安全上限。更大输入需要按地理网格切片，不进行隐式降分辨率。
- 原生单压缩块超过 64 MiB 拒绝解码，需要无损重分块。
- 单路密集采样最多 100000 点；超限时跳过可选轴线平滑，原始有效线保留，必要时生成明确的 C 级估计道路面。
- 区域可选走廊拟合超过 200 万证据点，跳过该可选拟合并记录警告，保留原区域道路；可能减少可选断裂补全，不声称这一分支与无限资源拟合等价。
- 精确骨架评价超过 800 万格时，该组指标标为未计算；真值辅助轴线网格也有分配前检查。尚未实现外存全局精确骨架算法，不能用粗网格分数冒充原精度。

RAM/VRAM 调度预算是保守预测与准入，不是操作系统强制 RSS 上限。GEOS 原生复杂几何、单期必要图层、Torch 驱动/工作区及外部程序仍会消耗内存。实际 MemoryError、CUDA OOM、磁盘/权限失败、依赖损坏、无法解释的 CRS/网格不一致及用户取消继续明确中止，不转成道路级成功。

## 测试结果

23 组针对性测试共 **460 次测试执行通过**。部分脚本复用测试类，此数字不代表 460 个互不重复场景。修改/新增 44 个 Python 文件通过语法编译，插件 145 文件后端快照校验通过，`git diff --check` 通过。

| 验证 | 结果 |
| --- | --- |
| 新增全流程极端测试 | 21 项通过：10 亿米有效线保留且 C 级回退、巨大评价网格不触发栅格分配、特征无损转存、密集拓扑分批、三进程资源竞争、串行退回后恢复预取、2048 条批导出及尾部回滚、40 个变化对流转、分组并集、局部证据、原算法走廊/覆盖数值对照、磁盘回溯、C 级与历史宽变屏蔽。 |
| RGB/容错/缓存 | 11 项 RGB 内存测试、15 项道路故障注入、33 项生产测宽、17 项宽度韧性、7 项缓存提交、10 项已有流水线内存安全测试通过；覆盖多线程/跨切片、真实系统错误中止及有效检查点复用。 |
| 正式成果/路网/变化/长时序 | 44 项生命周期、6 项修复接缝、23 项 Fast2 结构、21 项最终状态、17 项正式成果、12 项几何恢复、41 项退化路网、6 项面概化、48 项面质量、34 项当前后端合成回归通过。 |
| 预处理/并行及 GUI | 11 项 IR-MAD 缓存恢复、25 项处理优化；16 项插件、3 项警告 UI、35 项处理 UI、4 项评价展示通过。 |
| 已知巨型申请探针 | 再次运行审计探针；超长轴线/超大目标网格/巨大评价外包矩形均在危险分配前受保护，没有真实申请巨型数组。JSON 保留旧实现估算值时明确标注 historical。 |

真实数据及压力实测：

1. **4096×4096 GPU 推理对照**：RTX 4060 Laptop 8 GB，实际模型和已有规范化影像，一次旧实现/新实现对照。七项输出数组逐值相同（最大误差 0）。Torch 活动张量峰值从 3,833,285,632 降到 3,531,295,744 字节，减少 **288 MiB / 7.9%**；allocator reserved 峰值均为 4,747,952,128 字节，未下降。耗时 12.13 → 13.08 秒，增加 **7.8%**。没有把 RAM/磁盘转存宣称为 GPU 加速。
2. **已有项目 600 m 局部回归**：复用 plugtest 缓存，恢复后 55 条中心线、4 条宽度段、4 条走廊与改造前产物的几何 WKB 和共有属性完全一致。
3. **两期真实缓存下游完整链**：20221020、20230222 的 200 m 局部，运行区域恢复 → RGB 测宽 → 正式成果 → Fast2 → 最终协调 → 长时序。串行与自动并行的八个图层逐字段/逐几何相同。两期分别保留 7 条中心线、338/296 条宽度段；进程峰值 RSS **1.083 GiB**。串行 16.32 秒，自动 13.09 秒，单次约快 **19.8%**。未重跑提取模型，也未改写原缓存。此局部没有发布变化对象，非空变化另由合成回归覆盖。该测试非随机重复性能实验，不推断所有项目必然加速 20%。
4. **磁盘记录规模压力**：独立进程，每条约 500 字节负载，1 万 → 20 万条，峰值 RSS 26,099,712 → 26,124,288 字节（约 24.9 MiB，增加 24 KiB）；耗时 0.146 → 6.698 秒。这里只测记录模块，不代表全流水线 RSS。

初次扩展运行父仓库旧开发测试时发现其缺少当前质量等级/真实成果路径契约及导入路径配置，不能作为当前插件整套通过的证明。新增 `check_memory_backend_regressions.py` 复用其中有效数值场景，将已知合成测量宽度明确标为 A，34 项通过；没有放宽生产代码对缺失质量/C 级宽度的拒绝。当前插件自己的正式成果与真实缓存链另行验证。

## 复现命令

在 `road_change_plugin` 目录运行，不需要 pytest：

```powershell
runtime/env/samroad_env/python.exe tests/test_full_pipeline_memory.py
runtime/env/samroad_env/python.exe tests/test_rgb_memory_safety.py
runtime/env/samroad_env/python.exe tests/test_width_local_exceptions.py
runtime/env/samroad_env/python.exe tests/check_memory_backend_regressions.py
runtime/env/samroad_env/python.exe tests/check_final_road_states.py
runtime/env/samroad_env/python.exe tests/check_irmad_cache_recovery.py
runtime/env/samroad_env/python.exe tests/check_memory_peak.py
runtime/env/samroad_env/python.exe tests/check_cached_memory_flow.py ../project/plugtest --output _work/cached-memory-flow.json
runtime/env/samroad_env/python.exe tests/check_existing_period.py ../project/plugtest
gui_env/python.exe tests/test_plugin.py
gui_env/python.exe -m unittest discover -s tests -p test_evaluation_summary.py
```

其余测试脚本名称、实际计数和耗时见结果 JSON。GPU 对照单独显式执行，使用现有单张缓存 TIFF，不运行全项目提取：

```powershell
runtime/env/samroad_env/python.exe tests/check_inference_memory.py <已有缓存影像.tif> --size 4096 --output _work/gpu-memory.json
```

## 尚未验证与剩余风险

没有重跑七期全项目模型推理或全域真值精度评价；本次真实验证为一张完整 4096 切片和两期缓存局部完整下游流程。另机曾出现的 NumPy 40.5 GiB 申请未取得原始 traceback、源数据网格/道路几何，因此不能声称逐行复现了该机器的原始错误。

本次修复已知的无界稠密采样、特征驻留、跨期记录累计、大广播、候选存储和矩形栅格化风险，但不承诺任意规模/任意拓扑在任意内存机器上必然成功。单个超复杂 GEOS 结果、必要的单期区域图层及少量跨期状态索引仍随输入增长；资源预测不能控制其他程序占用、GPU 碎片或原生库工作区。超限会按上述明确边界降级或失败，已经完成的有效检查点仍可续跑。超大精确骨架评价和超大区域可选走廊拟合仍有明确功能边界，见前文。

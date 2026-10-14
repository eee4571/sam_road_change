# 多部件道路长时序与缓存恢复验证

验证日期：2026-10-14。范围仅为独立插件。另一台电脑上的报错项目未在本机运行。

## 几何与身份

- 一个道路记录保留全部线性部件，GeometryCollection 递归提取线性成员。
- 不选择最长段，不跨空隙连接；仅节点计算对恰好相连、无分叉的线段使用 line_merge，存储几何不合并。
- 分叉、断开、闭环、不简单道路的 from_node / to_node 为空。
- 匹配使用各线段端点和各部件内部的长度加权方向，避免空端点均值 NaN 或把部件间空隙当作道路方向。
- 生命周期 road_id 继续使用原 track_id；不再用 RC 前缀判断是否属于已关联道路。
- 生命周期、观测、事件、SHP 往返、人工复核接受/拒绝以及复核后复用观测均有回归。

真实 fixture 来自 plugtest / 20240106 的道路状态，track_id 为 RCf9fbd785aef9fd34，包含 5 个线性部件。

## 真实缓存完整汇总

`check_cached_temporal.py` 直接读取原项目道路、状态、变化文件，将汇总写入独立验证目录。
禁止调用道路提取、输入准备、变化检测和道路重建；比较上游文件 SHA-256 和修改时间。

| 项目 | 期次 | 生命周期道路 | 观测 | 事件 | 校验未变化文件 | 复核后重建 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| plugtest | 7 | 9726 | 68082 | 10019 | 211 | 通过 |
| othertest | 2 | 1110 | 2220 | 239 | 31 | 通过 |

plugtest 包含 2 条真正多部件的观测，另有 105 个单部件 MultiLineString 容器。
全部状态表的 track_id 和观测几何逐条核对一致。最终 plugtest 汇总约 125 秒，复核重建约 23 秒。

复现（在插件根目录）：

```powershell
.\runtime\env\samroad_env\python.exe tests/check_cached_temporal.py --manifest ..\project\plugtest\_work\current\pipeline_result.json --output _work\temporal_validation
```

## 自动测试

| 测试 | 结果 |
| --- | --- |
| test_temporal_multipart.py | 8 通过 |
| test_temporal_recovery.py | 5 通过 |
| test_project_state.py | 24 通过（含中断后重选范围） |
| test_ui_processing.py | 36 通过 |
| test_workbench.py | 17 通过 |
| test_plugin.py | 16 通过 |
| check_final_road_states.py（含其导入的真实算法合成测试） | 21 通过 |
| 原工作台 test_temporal_road_analysis.py，导入插件后端 | 5 通过 |

额外运行 test_fixed_policy.py：3 通过、1 个既有失败。
`test_results_do_not_publish_internal_facets_or_auto_geopackage` 期望通用名称 width.png 出现在结果中，
现有 result_parser 未返回该项。使用 HEAD 原版 result_parser 重现相同失败；本次未修改该模块或相关断言。

## 恢复行为

- 新增 `rerun-temporal --pipeline-manifest PATH [--grid AREA]`，界面对应“局部重跑 → 仅重建长时序”。
- 完整任务或局部重跑中断后，均可重新选择局部范围；“继续”恢复上次范围和断点，重新选择并启动则使用新范围。
- 仅长时序恢复不清理已有发布目录，不重新发布道路/测宽/变化成果；只完成长时序和必要评价、发布。
- 完整任务续跑先检查已固化的最终道路和相邻期变化成果，不依据新算法版本废弃这些完成记录。
- 缺失 SHP 配套文件或道路状态时明确报错；已经进入最终成果阶段的任务不会因此自动转回推理。
- 单区域恢复不会把其他缺失长时序的区域标记为全部完成。
- Plugin API、Signals、原 CLI 参数和成果 manifest 结构保持不变；仅新增一个 CLI 操作。

# 测宽可靠性改造：运行与验收说明

## 实际执行路径与故障定位

插件 `Controller → Runner → code/user_pipeline.py → engine.fast_pipeline`，固定 `fast + raw_image`。
`width` 阶段整理切片中心线；`regional` 阶段调用 `raw_image_backend.measure_region`，随后构建正式道路。
`molra_centerline_width` 不是当前插件 RGB 测宽入口；它的断面采样和 `chain_width_calculator` 也进行了道路级隔离。
运行日志现在记录入口绝对路径、实际解释器、测宽实现摘要。请用这些信息核对故障电脑是否运行新代码。

本次用户确认异常发生在另一台电脑，本机没有该次 traceback，不能认定已经复现或定位那次退出码 1 的唯一根因。
已证实的中断路径包括：RGB Worker 将道路异常传播到整个区域；零宽道路在 Fast 导出被跳过；
正式宽度剖面映射异常会丢弃有效轴线；缺失完成标记和过期输入可能被误认作可复用成果。

## 运行与续跑

更新插件的 `code/`、相关 `plugin/` 文件后，选择“继续处理”。不删除项目缓存，不选择会清空任务的新建完整运行。
推理、概率图和源掩膜保留；切片清理结果写入测宽目录，不覆盖原始掩膜。
首次升级需要重新生成宽度及下游成果，但不因测宽代码升级而使道路提取模型缓存失效。
旧版本未保存的道路链/切片检查点不能追溯恢复。

在项目 `project_config.json` 顶层可设置 `"width_default_m": 6.0`（米，必须有限且大于零）。
插件通过项目环境读取该值；直接使用后端函数可传 `default_width_m`。
默认宽度是 C 级临时占位，不是实测值。参数或实现变化会使测宽缓存失效。

回退顺序：正常测量 → 同一道路可靠断面插值/中位数 → 连续链可信邻接值 →
局部掩膜几何/距离变换 → 区域可靠中位数 → 项目默认。
可靠邻接只取原始可信供体，不让回退值递归充当证据。所有宽度在米制坐标中验证；不裁切或拉伸影像以匹配矢量。
部分/全段越界保持原始中心线，仅限制证据可用性；非法、非有限、零长度几何单独诊断。
RGB 正常测量和重建公式保持不变。超出单道路 20000 个断面的计算预算时转为有诊断的低可信回退。

## 状态、输出与诊断

- `width_review/tile_checkpoints/`：逐切片 GPKG 和原子完成标记。
- `products/network_checkpoint/`：独立路网恢复检查点。
- `products/raw_width/chains/<输入身份>/`：逐道路链观测及完成标记；未完成临时文件不复用。
- `products/raw_width/preflight.json`：输入几何快速检查。
- `products/raw_width/summary.json`、`products/width_quality_summary.json`：数量、来源、质量、长度、道路 ID、切片 ID、定位范围/CRS、异常类型、堆栈、回退结果。
- `products/width_export_summary.json`：正式成果构建后的质量汇总，包含剖面映射和宽度段构建的局部回退。
- `products/raw_width/failure.json`：系统失败时的上下文；成功重试后移除旧失败记录。
- GPKG 保留 `width_source`、`quality_grade`、`width_quality`、`outlier_reason`、`qa_state` 等字段；公开 SHP 用短字段 `width_qual` 等传递可信度。

局部可恢复异常产生 `completed_with_warnings` 质量状态；为了保留现有任务/发布接口，任务本身仍是成功完成，
结果清单补充 `width_status`、`width_warnings`，并发送结构化 warning 事件。
GUI 提示“测宽完成，部分道路使用估计宽度”，日志中的“查看技术详情”直接读取质量报告。
C 级和未知质量宽度不会进入 Fast2 自动拓宽/变窄剖面，也不会进入宽度偏差校准；道路存在性成果继续处理。
将混合 A/B/C 的整链代表宽度保守标为 C，可能降低低证据区域的自动宽变召回，这是明确的可靠性取舍。
权限、磁盘/文件读写、内存、依赖/模型环境、无法解释的 CRS/栅格契约等错误继续失败；没有强制返回 0。

## 可执行验证

在插件根目录运行，均无需模型推理：

```powershell
runtime/env/samroad_env/python.exe tests/test_width_production.py -v
runtime/env/samroad_env/python.exe tests/check_width_resilience.py -v
runtime/env/samroad_env/python.exe tests/check_fast2_structure.py -v
runtime/env/samroad_env/python.exe tests/check_formal_period_products.py -v
runtime/env/samroad_env/python.exe tests/check_processing_optimizations.py -v
gui_env/python.exe tests/test_width_warning_ui.py -v
gui_env/python.exe -m unittest discover -s tests -p 'test_ui_processing.py' -v
```

覆盖普通线、短线、重复/零长度、自交、闭环、分叉、孤立线、部分/全部越界、空/nodata 掩膜、
形状错误、NaN/Inf、异常索引、5000 条道路、单道路失败、单切片失败、系统中断续跑、损坏检查点、
参数失效、原始测量数值保持、GPKG 质量字段、正式道路保留和低可信 Fast2 限制。
真实子进程测试验证：道路警告退出码 0；栅格契约错误退出码非零且没有完成标记。

2026-10-12 的具体结果见 `WIDTH_TEST_REPORT.md`。

## 尚未验证的范围

没有复现另一台电脑的原始失败，也没有运行完整长时序/模型推理。full 模式本次只验证了共享采样、链计算及几何函数，
未做 full 历史流水线端到端验收。本次真实数据检查是一个项目的限定空间子集，不等于所有区域精度评估。
仍需故障电脑的完整运行日志、`period_state.json`、`input_manifest.json`、实际代码版本和失败切片的
中心线/掩膜/影像元数据；若新报告已有 `failure.json` 或 `summary.json`，一并提供即可先判断是否需要影像像素。

# 道路变化检测 QWidget 独立插件

版本 1.1.0，Plugin API 1 / Signals / QWidget 宿主接口保持不变。

## 独立后端

`code/` 是当前主工作台正式后端的完整生产依赖副本，直接复制、不从主工作台动态导入或启动主工作台进程。
插件运行路径：

QWidget → Controller → Runner → 插件 runtime/env/samroad_env/python.exe → 插件 code/user_pipeline.py。

PYTHONPATH、模型、GIS 数据库和子进程全部使用插件目录。主工作台可以移走，插件代码不引用其目录。
外部影像和用户选择的成果目录仍按输入路径使用。

仅在开发同步时执行 `python tools/sync_backend.py <主工作台目录>`。
该工具覆盖当前生产副本并移除已废弃的模块，生成 `resources/backend_snapshot.json` 的逐文件 SHA-256。
运行时不会执行同步工具，也不需要主工作台或其 experiments/tests/_profiling 目录。
源树中的共用类/函数按原样复制；独立 baseline 不作为插件运行选项或 fallback。

## 固定正式配置

`resources/production_config.json`：

- 正式 Fast2 流程；无 Full/Fast1 入口。
- Fast IR-MAD 开启，参考期 20250118。各目标期独立拟合，参考期使用原图。
- IR-MAD 迭代最多 4,000,000 个确定性抽样像元；最终 NCP、PIF > 0.95、TLS 仍使用全部有效像元。
- 最终测宽仅使用原始影像边界后端，含异常剔除、连续重建、代表宽度和规则展示面。
- 候选影像局部辐射校正关闭；其余三项补偿开启。
- 原始细粒度测宽数据继续用于 Fast2，规则展示面不替代分析证据。

命令边界统一应用固定配置，忽略旧模式/测宽/补偿参数覆盖。任务日志记录配置；正式后端将其写入 input_spec、诊断和缓存身份。
完整流程仍为：影像规范化 → 各目标期 IR-MAD → 道路提取和区域后处理 → 原始影像测宽/道路面 → Fast2 → 有真值时后验处理 → 最终道路/变化/长时序 → 有真值时评价。

SAM-MoLRA 仍提供主工作台现有区域恢复所需的辅助证据，因此模型必须安装；它不再是插件可选择的最终测宽方法。

## 安装资源

```
runtime/env/samroad_env/          完整、可移动的算法 Python 环境
runtime/model/samroad/samroad.ckpt
runtime/model/samroad/sam_vit_b_01ec64.pth
runtime/model/sam_molra/adapter.th
runtime/model/sam_molra/sam_vit_b_01ec64.pth
runtime/config/samroad_inference.yaml
```

当前本地验证已复制独立环境和模型，源码版本控制仍忽略这些大文件。打包交付时需携带上述目录。
宿主环境只需 PySide6；不导入 torch、GIS、Tkinter 或算法模块。`start_plugin.bat` 和 `standalone.py` 保留。

## 宿主接口

```python
from plugin import create_plugin
plugin = create_plugin()
widget = plugin.create_widget(parent)
host_layout.addWidget(widget)
plugin.result_ready.connect(on_result)
# 宿主退出时
plugin.shutdown()
```

宿主拥有 QApplication；create_widget 不创建应用或独立顶层窗口。
信号均为 dict：task_started、task_progress、task_log、task_finished、task_failed、result_ready。
结果类型维持 road_centerline、road_surface、road_width、road_change、road_temporal、road_evaluation。
正式目录沿用 01_单期道路 / 02_变化检测 / 03_长时序；内部配置和审核字段不另行公开。

## 续跑、局部重跑和失败

同配置完整任务可正常续跑；期次重跑更新相邻变化和长时序，变化重跑更新关联最终成果。
旧 Raw/SAM-MoLRA/Full 任务允许查看，不自动改写为当前配置。局部重跑发现上游配置不符时明确报错，要求新建完整任务，避免误用旧道路缓存。
部分失败、子进程异常、取消维持原信号；失败不会误报成功。

## 验证

轻量插件测试：`python -m unittest discover -s tests -p "test_*.py"`（PySide6 环境，无模型推理）。
复制后端合成回归：用插件算法解释器执行 `tests/run_backend_smoke.py <主工作台/code/tests>`；测试源码仅为开发输入，所有被测后端必须来自插件副本，生产不依赖测试源码。
`tests/run_real_pipeline.py` 是显式手动真实任务驱动，不在 unittest discovery 中；未经要求不运行。

本轮真实任务尝试因长路径图片写入失败而退出，未完整完成。后续按用户要求停止真实模型验证；已同步长路径图片 I/O 修复并通过微型图像读写测试，未重新跑真实任务。

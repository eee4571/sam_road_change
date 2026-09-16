# 道路变化检测 QWidget 独立插件

版本 1.1.0，Plugin API 1 / Signals / QWidget 宿主接口保持不变。

## 独立后端

`code/` 是当前主工作台正式后端的完整生产依赖副本，直接复制、不从主工作台动态导入或启动主工作台进程。
插件运行路径：

QWidget → Controller → Runner → 插件 runtime/env/samroad_env/python.exe → 插件 code/user_pipeline.py。

PYTHONPATH、模型、GIS 数据库和子进程全部使用插件目录。主工作台可以移走，插件代码不引用其目录。
外部影像和用户选择的成果目录仍按输入路径使用。

两套源代码独立维护，不再提供自动覆盖工作台代码的同步脚本。`resources/backend_snapshot.json` 仅记录本插件自己的 SHA-256 校验值。

## 固定正式配置

`resources/production_config.json`：

- 正式 Fast2 流程；无 Full/Fast1 入口。
- Fast IR-MAD 开启，参考期由用户在数据页为每个验证区独立选择并保存。各目标期独立拟合，参考期仅使用统一网格原影像。
- IR-MAD 迭代最多 4,000,000 个确定性抽样像元；最终 NCP、PIF > 0.95、TLS 仍使用全部有效像元。
- 最终测宽仅使用 RGB影像边界后端（IR-MAD 后影像），含异常剔除、连续重建、代表宽度和规则展示面。
- 候选影像局部辐射校正关闭；其余三项补偿开启。
- Fast2 与规则道路面使用路口间稳健代表性宽度；细粒度观测与逐点 profile 仅保存在内部 GPKG。

命令边界统一应用固定配置，忽略旧模式/测宽/补偿参数覆盖。任务日志记录配置；正式后端将其写入 input_spec、诊断和缓存身份。
完整流程仍为：影像规范化 → 各目标期 IR-MAD → 道路提取和区域后处理 → 原始影像测宽/道路面 → Fast2 → 有真值时后验处理 → 最终道路/变化/长时序 → 有真值时评价。

RGB 正式流程完全跳过 SAM-MoLRA 加载、推理与旧测宽，不需要安装 MoLRA 模型。区域构造在独立 regional 阶段完成，export 只写成果。

## 安装资源

```
runtime/env/samroad_env/          完整、可移动的算法 Python 环境
runtime/model/samroad/samroad.ckpt
runtime/model/samroad/sam_vit_b_01ec64.pth
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

## 当前项目、继续处理与局部重跑

每个项目只管理一套当前正式成果，以及最多一个未完成处理状态。用户不需要选择任务编号；运行记录只展示当前日志和最近错误。

```
项目/
├─ project_config.json
├─ _work/
│  ├─ current/             当前处理、断点及中间结果
│  ├─ cache/               由后端按原有效性规则复用的缓存
│  ├─ project_state.json   当前操作状态及局部更新范围
│  └─ current_results.json 当前可用成果路径
├─ 成果输出/               当前正式成果
└─ _logs/plugin_ui/current.log
```

- **运行完整流程**：先检查参数和运行资源，再清理旧正式成果及其绑定工作目录，重新开始。保留输入数据、项目配置和通用缓存。
- **继续未完成处理**：打开项目时自动发现断点，存在时才显示按钮；完整流程使用原后端续跑参数。取消的局部重跑会继续执行原区域和原期次/变化对的重跑命令，不要求用户重新选择。
- **局部重跑**：直接使用当前项目成果。期次重跑清理该期、按原相邻证据依赖确定的变化对和后续汇总；变化对重跑清理该对和相关汇总。发布层保留无关区域、期次与变化对的正式文件。
- **打开旧项目**：只读取原“最近成果”索引作为当前成果，不枚举历史任务。再次完整运行后使用新目录并移除旧工作目录。

运行前会校验清理范围，拒绝清理输入数据、通用缓存、其他项目或越界链接。旧处理配置不兼容时要求重新运行完整流程。Plugin API、六个 Signals、result_type 和算法处理链不变。

生命周期轻量验证：`python -m unittest discover -s tests -p test_project_state.py`。使用临时文件和模拟 Runner，覆盖首次运行、取消后继续/重启、成果替换、两类局部重跑及缓存保留，不运行模型。

## 测宽缓存与验证

RGB/Lab/灰度按块缓存；Gaussian/Sobel/Canny/texture 按完整窗口精确缓存，保留 Canny 的窗口连接语义。共用 128 MiB LRU 上限，最多 4 个 road-chain 求解线程，不写整区 RGB 拼接图。

## 验证

轻量插件测试：`python -m unittest discover -s tests -p "test_*.py"`（PySide6 环境，无模型推理）。
复制后端合成回归：用插件算法解释器执行 `tests/run_backend_smoke.py <主工作台/code/tests>`；测试源码仅为开发输入，所有被测后端必须来自插件副本，生产不依赖测试源码。
`tests/run_real_pipeline.py` 是显式手动真实任务驱动，不在 unittest discovery 中；未经要求不运行。

本轮真实任务尝试因长路径图片写入失败而退出，未完整完成。后续按用户要求停止真实模型验证；已同步长路径图片 I/O 修复并通过微型图像读写测试，未重新跑真实任务。

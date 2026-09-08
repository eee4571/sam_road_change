# 道路变化检测 QWidget 插件

独立交付目录，可整体移动。现有 SamRoadChange 源码未修改；`code/` 是正式后端的独立副本，包含 user_pipeline、应用层和算法依赖。未复制 Tkinter 页面、编辑器管理器、训练数据、项目成果、环境或模型。正式后端内部用于结果固化的函数保持原样，插件不提供人工编辑或人工复核入口。

```text
road_change_plugin/
├─ plugin/
│  ├─ __init__.py
│  ├─ plugin.py          # Plugin API 与生命周期
│  ├─ widget.py          # 四页 QWidget
│  ├─ controller.py      # 输入检查、用户意图、命令与成果协调
│  ├─ runner.py          # QProcess 与结构化协议
│  ├─ signals.py
│  ├─ result_parser.py   # 正式 manifest → 稳定成果类型
│  └─ ui/               # 紧凑表单、折叠区、数据表格
├─ code/                # 95 个正式 Python 后端源文件
├─ runtime/
│  ├─ config/samroad_inference.yaml
│  ├─ env/PLACEHOLDER
│  └─ model/PLACEHOLDER
├─ resources/backend_snapshot.json
├─ tests/test_plugin.py
├─ plugin.json
├─ standalone.py
└─ README.md
```

## Plugin API 1

宿主 Python 安装 PySide6，并由宿主创建 QApplication。宿主加载插件目录后：

```python
from plugin import create_plugin

plugin = create_plugin()
widget = plugin.create_widget(parent)
host_layout.addWidget(widget)
plugin.result_ready.connect(on_result)
# 宿主退出时：
plugin.shutdown()
```

宿主持有 `plugin` 引用。`create_widget()` 仅返回 QWidget，不创建 QApplication、QMainWindow 或 QDialog。关闭 QWidget 不取消任务；`shutdown()` 停止本插件持有的后台进程。Windows 下取消会终止该后端的子进程树。目标运行平台为当前项目使用的 Windows；其他系统的算法环境和子进程树取消未验证。

属性：`plugin_id="road_change"`、`name="道路变化检测"`、`version="1.0.0"`、`api_version="1"`。

所有公开信号均为 `Signal(dict)`：

| Signal | 字段 |
|---|---|
| task_started | task_id、plugin_id、name |
| task_progress | task_id、plugin_id、progress（0~1）、stage、message、event |
| task_log | task_id、level、message；结构化日志另含 event |
| task_finished | task_id、plugin_id、status（completed/cancelled）、exit_code、event |
| task_failed | task_id、plugin_id、message、detail、status |
| result_ready | task_id、plugin_id、result_type、name、path、metadata |

输入检查失败的 task_id 为空；已有成果浏览的 task_id 为空。后端阶段事件的进度是当前阶段或工作单元进度，可能随阶段切换重新计数。完成信号在进程退出后发送；部分失败、协议错误、异常退出通过 task_failed 报告。普通日志只展示，不决定核心状态。

## 四个 UI 页面

- 数据准备：区域名称与 SHP、每区域多期影像 TXT、可选相邻期 GT、评价开关、成果目录。表格“选择文件”填写当前行的路径列，其他列直接编辑；区域名称须一致。
- 运行处理：标准 / Fast、完整流程、任务名称和续跑；“局部运行与重跑”默认折叠，支持某期及其相关成果更新、某变化对及长时序更新；高级参数默认折叠；运行资源检查与取消。
- 成果与评价：读取正式 `pipeline_result.json`，展示 road_centerline、road_surface、road_width、road_change、road_temporal、road_evaluation；可对已有成果运行正式评价命令。只通过 result_ready 报告现有文件路径，不访问宿主地图。未最终固化的 Fast 成果不报告。
- 运行记录：当前会话日志与 JSON 事件，可导出记录；界面保留最近 2000 行。

默认继承宿主 Qt Style、Palette、字体和 QSS。无全局主题设置和固定窗口尺寸，表单按行折叠，表格支持水平滚动，适配约 300px 及以上面板。

## Runner 调用

```text
QWidget → Controller → Runner → QProcess
  → runtime/env/samroad_env/python.exe -u code/user_pipeline.py <参数>
```

运行时从 `plugin/runner.py` 的 `__file__` 定位插件根目录，工作目录为插件 `code/`，不依赖宿主工作目录。QProcess 使用参数列表，不经过 shell。子进程设置 UTF-8、插件 code 的 PYTHONPATH、独立环境 DLL/PROJ/GDAL 路径，以及 `SAMROAD_MODELS_ROOT=<插件根>/runtime/model`。宿主进程不导入后端或 GIS/模型依赖。

映射：完整运行 → `all --mode validation --execution-profile full|fast`；期次重跑 → `rerun-period --update-related`；变化对重跑 → `rerun-change --update-temporal`；已有成果评价 → `evaluate-all-existing`。命令参数沿用正式后端；相对输入路径以插件根目录解析，用户选择的外部输入和输出路径按原位置使用。

优先解析 `__SAMROAD_USER__{JSON}`，支持分块 UTF-8、跨读取分行、末行无换行。保留正式调度与算法链：单期道路 → Auto 变化 → 可选 GT 后验校正 → Final Roads → Final Changes → Final Temporal → Final Evaluation；具体阶段和可选评价条件由复制的正式后端按标准/Fast 模式执行。

## 手动放置环境与模型

当前 env 和 model 中只有 PLACEHOLDER。把可独立运行的完整环境内容放入：

```text
runtime/env/samroad_env/
├─ python.exe
├─ Lib/site-packages/
├─ Library/bin/
└─ ...完整环境其余文件
```

模型按以下位置放置，文件名保持一致：

```text
runtime/model/
├─ samroad/
│  ├─ samroad.ckpt
│  └─ sam_vit_b_01ec64.pth
└─ sam_molra/
   ├─ sam_vit_b_01ec64.pth
   └─ adapter.th
```

标准模式检查四个权重文件；Fast 检查 samroad 的两个权重。必须复制可迁移的完整环境，不能只复制 python.exe。模型不会自动下载。正式推理配置已在 `runtime/config/`，SAMRoad 运行时通过模型根目录定位权重。

## standalone 与本轮验证

在插件目录使用带 PySide6 的界面 Python 执行 `python standalone.py`。算法始终使用上述独立环境。

Windows 下也可双击 `start_plugin.bat` 一键打开。启动器依次检查系统 `python`、`py` 和插件 `runtime/env/samroad_env/python.exe`，选择可以导入 PySide6 的解释器；启动失败时保留错误窗口。无需先安装模型即可打开界面。执行 `start_plugin.bat --check` 可仅检查界面启动环境。

测试：`python -m unittest discover -s tests -v`。覆盖入口、standalone 生命周期、页面切换、300px 布局约束、命令构造、缺失资源、协议、模拟 QProcess 完成/失败/取消、结果路径、迁移与导入边界。测试创建的子进程仅输出模拟文本，不执行正式后端或模型。`resources/backend_snapshot.json` 记录逐文件 SHA-256，可核对后端副本未改动。

本轮没有执行真实模型或真实项目完整流程；安装环境、权重后的真实推理与端到端结果仍需后续验证。

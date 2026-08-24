# SamRoadChange Code Map

本文件用于帮助开发工具快速定位代码。

目标不是记录所有文件，而是避免为了寻找入口而扫描整个仓库。

---

## 1. GUI 启动关系

```text
launcher / 启动脚本
        ↓
code/user_workflow_gui.py
        ↓
code/gui/main_window.py
        ↓
UserApp
 ├─ DataPage
 ├─ RunPage
 ├─ EditPage
 └─ ResultPage
```

`user_workflow_gui.py` 是兼容入口。

真正 GUI 实现在 `code/gui/`。

---

## 2. GUI 页面

### 数据准备

```text
code/gui/data_page.py
        ↓
code/app/project_manager.py
        ↓
code/input_catalog.py
```

主要涉及：

- 新建/打开项目
- 外部数据源连接
- 验证区
- 多期影像 TXT
- 真值
- 数据扫描
- 数据检查
- 项目配置

修改数据准备 UI 时通常不需要读取算法代码。

---

### 自动处理与局部重跑

```text
code/gui/run_page.py
        ↓
code/gui/main_window.py
        ↓
code/app/task_manager.py
        ↓
code/app/backend_client.py
        ↓
code/user_pipeline.py
```

`run_page.py` 主要负责界面。

用户意图由 `TaskManager` 转换为后端命令。

主要任务包括：

```text
完整运行
期次重跑
期次重跑 + 更新相关结果
变化对重跑
变化对重跑 + 更新长时序
批量道路提取
运行前检查
任务取消
```

如果只是：

- 改按钮位置
- 改文字
- 改区域/期次/变化对选择器
- 调整卡片布局

通常只需要修改 `run_page.py`。

如果改变按钮执行逻辑，再查看 `main_window.py` 和 `task_manager.py`。

---

## 3. 后端进程边界

```text
GUI
 ↓
BackendClient
 ↓ subprocess
user_pipeline.py
```

文件：

```text
code/app/backend_client.py
```

负责：

- 启动 Python 后端
- stdout
- 普通日志
- 结构化事件
- priority queue
- cancel
- 环境变量
- Python executable

结构化后端事件前缀：

```text
__SAMROAD_USER__
```

如果只是 GUI 页面修改，不需要读取 `user_pipeline.py`。

---

## 4. 任务管理

```text
code/app/task_manager.py
```

负责用户任务意图和后端 CLI 参数之间的转换。

重点能力：

```text
完整 pipeline 命令
自动判断新任务/续跑
未完成任务检测
期次重跑
变化对重跑
受影响相邻变化对计算
批量道路提取
人工编辑应用
已有成果精度评价
```

修改“续跑/重跑应该怎么工作”时优先看这里。

不要首先进入算法目录。

---

## 5. 项目管理

```text
code/app/project_manager.py
```

负责：

```text
project_config.json
数据源扫描
验证区和期次配置
真值配置
结果索引
成果预览
人工复核成果索引
长时序成果索引
```

项目扫描已经排除：

```text
.git
env
.venv
venv
__pycache__
04_成果输出
_work
_logs
models
runtime
cache
tmp
...
```

因此不要另外设计一套全仓库扫描逻辑。

---

## 6. 结果目录与发布

```text
code/app/result_publisher.py
```

主要关注：

- `ProjectLayout`
- pipeline 输出目录
- period result
- change result
- latest result
- pipeline manifest
- GUI 结果索引

如果任务涉及成果路径或 manifest，先看这里。

---

## 7. 人工编辑

```text
code/gui/edit_page.py
        ↓
code/gui/main_window.py
        ↓
code/app/editor_manager.py
```

编辑完成后的正式成果更新：

```text
TaskManager
        ↓
apply-edits
        ↓
user_pipeline.py
        ↓
受影响切片重建
        ↓
重新测宽
        ↓
期次产品更新
        ↓
受影响变化对更新
        ↓
长时序更新
```

如果只是人工编辑页面布局，不需要读取 pipeline。

---

## 8. 结果与精度评价

```text
code/gui/result_page.py
        ↓
code/app/project_manager.py
code/app/task_manager.py
code/app/result_publisher.py
```

涉及：

- 中心线成果
- 道路面成果
- 宽度成果
- 变化成果
- review 成果
- 长时序道路属性
- 精度评价

---

## 9. 后端 pipeline

```text
code/user_pipeline.py
```

这是大型后端调度文件。

当前主要期次阶段：

```text
centerline
  道路提取

surface
  道路面提取

width
  道路宽度计算

finalize
  结果固化

export
  道路产品导出
```

随后执行：

```text
相邻期变化检测
↓
长时序分析
```

不要为了 GUI 修改全文读取本文件。

需要修改某阶段时，先搜索具体 command、函数或 stage，再读取相关代码段。

---

## 10. 算法模块

### 道路中心线

```text
code/engine/samroad/
```

SAMRoad 和相关中心线算法。

---

### 道路面

```text
code/engine/sam_molra/
```

SAM-MLoRA 及道路面相关处理。

---

### 道路宽度

```text
code/engine/width/
```

道路宽度计算及相关融合处理。

---

### 长时序

```text
code/temporal_road_analysis.py
```

跨期稳定道路对象、生命周期和事件分析。

---

## 11. 常见需求 → 首选文件

| 需求 | 首先读取 |
|---|---|
| 修改数据页布局 | `gui/data_page.py` |
| 修改运行页布局 | `gui/run_page.py` |
| 修改人工编辑页 | `gui/edit_page.py` |
| 修改成果页 | `gui/result_page.py` |
| 修改全局窗口布局 | `gui/main_window.py` |
| 修改按钮样式/间距 | `gui/common_widgets.py` + 对应页面 |
| 修改项目扫描 | `app/project_manager.py` |
| 修改续跑逻辑 | `app/task_manager.py` |
| 修改局部重跑依赖 | `app/task_manager.py` |
| 修改后台进程通信 | `app/backend_client.py` |
| 修改成果目录 | `app/result_publisher.py` |
| 修改 pipeline 行为 | `user_pipeline.py` |
| 修改道路中心线算法 | `engine/samroad/` |
| 修改道路面算法 | `engine/sam_molra/` |
| 修改测宽算法 | `engine/width/` |
| 修改长时序 | `temporal_road_analysis.py` |

---

## 12. 默认读取策略

例如任务：

> 将运行页“局部重跑”重新排版。

读取：

```text
AGENTS.md
docs/CODEMAP.md
code/gui/run_page.py
```

如果只是布局修改，到这里即可。

不要读取：

```text
user_pipeline.py
engine/
```

---

任务：

> 重跑某期后自动重跑前后两个变化对。

读取：

```text
AGENTS.md
docs/CODEMAP.md
code/app/task_manager.py
main_window.py 中调用位置
相关测试
```

只有发现后端缺少对应能力时：

```text
再读取 user_pipeline.py 中 rerun-period / rerun-change 相关部分
```

---

任务：

> 优化 SAMRoad TopoNet。

直接进入：

```text
engine/samroad/
相关 pipeline 调用位置
相关测试
```

此时不需要读取四个 GUI 页面。

---

原则：

**先根据任务找到最小修改边界，再读取代码；不要先扫描仓库再决定改什么。**
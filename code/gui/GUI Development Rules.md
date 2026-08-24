# GUI Development Rules

本文件适用于 `code/gui/` 下的修改。

## GUI 职责

GUI 负责：

- 页面布局
- 控件
- 用户输入
- 状态展示
- 任务进度
- 日志
- 成果展示

GUI 不负责实现：

- SAMRoad 算法
- SAM-MLoRA 算法
- 道路宽度算法
- 变化检测算法
- 长时序算法

---

## 当前结构

主窗口：

```text
main_window.py
```

`UserApp` 组合：

```text
DataPage
RunPage
EditPage
ResultPage
```

页面文件主要用于构建各自 UI。

一些 callback 和共享状态仍由 `UserApp/main_window.py` 提供，因此看到：

```python
command=self.some_method
```

而当前页面没有定义该方法时，不要假定代码缺失。

先在 `main_window.py` 查找对应方法。

---

## GUI-only 修改规则

如果任务只涉及：

- 控件大小
- 页面比例
- 布局
- padding
- 字体
- 按钮顺序
- 卡片顺序
- 文案
- 显示/隐藏
- 界面流程表达

默认只修改：

```text
对应 *_page.py
common_widgets.py（需要时）
main_window.py（全局结构需要时）
```

不要读取或修改：

```text
../user_pipeline.py
../engine/
```

---

## 行为修改

如果按钮行为需要改变：

```text
GUI page
↓
main_window.py callback
↓
../app/*_manager.py
```

优先修改或复用 manager。

不要直接从 GUI 调用：

```text
engine/samroad
engine/sam_molra
engine/width
```

---

## UI 设计原则

这是面向 GIS/遥感生产人员的专业桌面软件。

优先：

- 清晰
- 稳定
- 紧凑
- 信息层级明确
- 类似专业 GIS 桌面软件

避免：

- 大量解释性文字
- AI 产品式大卡片
- 重复状态模块
- 同一操作多个入口
- 暴露内部算法术语
- 暴露不必要的“断点/缓存/内部状态”概念

用户表达“想做什么”，软件决定后台如何执行。

例如优先：

```text
运行完整流程
重跑该期
重跑并更新相关结果
重跑该变化对
```

而不是暴露：

```text
resume
continue cache
stage recovery
partial pipeline
```

---

## 当前四步流程

保持：

```text
① 数据准备
↓
② 自动处理
↓
③ 人工编辑（可选）
↓
④ 成果与评价
```

除非任务明确要求，不改变这四步的产品结构。

---

## 修改完成后

GUI-only 修改只需进行对应的轻量检查。

不要为了验证一个布局修改启动完整 SAMRoad 推理流程。
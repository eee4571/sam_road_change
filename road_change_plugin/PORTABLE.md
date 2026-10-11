# Windows 便携包使用说明

适用于 Windows 10/11 64 位（x64）。完整解压 ZIP 后双击 `start_plugin.bat`。
无需安装系统 Python、PySide6 或联网下载依赖。

## 目录与已有 runtime

```text
road_change_plugin/
  start_plugin.bat
  standalone.py
  gui_env/                       独立界面环境，随包提供
    python.exe                   Python 3.14.3 嵌入式发行版
    Lib/site-packages/           PySide6 Essentials / shiboken6 6.11.2
  plugin/                        界面及业务协调代码
  code/                          后端代码
  experiments/                   后端相关配套源码
  resources/                     界面资源及正式配置
  runtime/
    config/samroad_inference.yaml 随包提供的小型算法配置
    env/samroad_env/python.exe    使用目标电脑已有的算法环境，不在 ZIP 中
    model/samroad/                使用目标电脑已有的模型，不在 ZIP 中
```

目标电脑如果已有 `road_change_plugin/runtime`，将压缩包解压、合并到该
`road_change_plugin` 目录即可。更新时先退出插件，不要删除已有 runtime。
如果 runtime 在其他位置，将已有的 `env`、`model` 目录放到以上位置；注意
模型目录名为 `model`，不要多套一层 `runtime`。后端仍要求其原有环境可移动、
依赖完整，GPU 驱动等系统条件仍沿用原后端要求。

即使没有 runtime 的算法环境和模型，也可直接打开界面；执行处理前才会检查它们。
本包不包含输入影像、项目数据、处理成果或日志。

`gui_env` 与 `runtime` 完全分开，不要把 PySide6 安装到算法环境。
随包保留 Python 和 Qt wheel 的许可文件、原生 DLL 及 Qt 平台插件。
Essentials 包提供界面用到的 QtCore、QtGui、QtWidgets，不包含未使用的 Addons。
构建会移除本 QWidget 界面不使用的 QML 资源及开发头文件，避免开发文件的超长
路径导致 Windows 解压失败；原生运行库、Qt 插件及许可文件保留。

## 自检与重新打包

在插件目录运行 `start_plugin.bat --check`，会实际创建并关闭界面控件，输出
所用 Python 路径；此检查不执行模型、不验证算法环境。正常双击则打开界面。

开发电脑执行：

```powershell
python road_change_plugin/tools/build_portable.py
```

构建时需要 Git 和联网下载官方 Python 嵌入式 ZIP、PyPI wheel；固定版本与
SHA-256 校验值保存在构建脚本中。构建不调用 pip，不修改系统环境或 runtime。
下载缓存放在 `build/downloads`，独立环境放在 `gui_env`，压缩包和校验文件放在
`dist`。后续可复用下载缓存和同版本 gui_env。

构建会将 ZIP 解压到含中文和空格的新目录，移除 PATH 中的系统 Python/Qt，
检查文件校验值、导入隔离、后台解释器路径、Qt 事件循环与 BAT 启动。
验证不运行真实遥感流水线，也不能代替目标机器的 GPU/算法环境验证。

`portable_manifest.json` 记录交付文件 SHA-256；ZIP 旁的 `.sha256` 文件用于
传输后检查完整性。若开发中修改界面依赖，请同步调整固定 wheel 并重新构建，
不要将本机 venv 直接复制进便携包。

官方来源：[Python 3.14.3](https://www.python.org/downloads/release/python-3143/)、
[Qt for Python](https://doc.qt.io/qtforpython-6/gettingstarted.html)。

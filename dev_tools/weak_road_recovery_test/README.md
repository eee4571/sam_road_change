# Weak Road Recovery Test

推荐方式：双击 `launcher.pyw`。

选择影像 → 选择 CUDA/CPU、Batch Size 和 Threshold Profile → 点击“完整运行” → 查看
`recovery_compare.png`。选择影像后，输出目录会自动设为
`outputs/<影像名>/`，也可以手动修改。

后续调参时，展开“高级参数”，修改 Low Threshold、Max Gap 等参数，
再点击“只重新弱恢复”。程序会复用第一次运行保存的概率图和原始 graph，
不重新加载或运行 SAMRoad，通常几秒即可得到新的对比结果。

对于整景概率偏低的影像，可以选择 `weak_sensor` profile，并保持
“Enable Bootstrap”开启。界面会显示 Scene Confidence 和 Recommended Profile。
Bootstrap 参数只影响本次测试，不会写回正式 YAML。

运行日志、状态和恢复统计会实时显示在窗口中。“打开结果目录”和“打开对比图”
可直接查看产物。最近一次路径、设备、Batch Size 和高级参数会保存在本地
`launcher_settings.json`，该文件已被 Git 忽略。

主要诊断产物：

- `scene_probability_diagnostic.png`：原图、概率热力图、HIGH mask、LOW mask。
- `bootstrap_overlay.png`：黄色 strong SAMRoad、青色 weak recovered、品红色 weak bootstrap。
- `recovery_compare.png`：原始 strong graph 与最终三来源 graph 对比。

## 命令行备用方式

首次运行 SAMRoad 并保存恢复前缓存：

```powershell
python dev_tools/weak_road_recovery_test/run_test.py --image D:/test/test_4096.tif
```

只重新运行 weak recovery，不重新加载模型：

```powershell
python dev_tools/weak_road_recovery_test/run_test.py --recovery-only --run-dir dev_tools/weak_road_recovery_test/outputs/test_4096
```

常用调参示例（参数只影响本次运行，不写回正式 YAML）：

```powershell
python dev_tools/weak_road_recovery_test/run_test.py --recovery-only --run-dir dev_tools/weak_road_recovery_test/outputs/test_4096 --road-low-threshold 0.16 --max-gap 80 --min-mean-probability 0.18
```

低置信度整景诊断与 bootstrap 示例：

```powershell
python dev_tools/weak_road_recovery_test/run_test.py --recovery-only --run-dir dev_tools/weak_road_recovery_test/outputs/test_4096 --threshold-profile weak_sensor --enable-bootstrap --bootstrap-min-length 48 --bootstrap-min-background-contrast 0.08
```

黄色为原始 SAMRoad 中心线，青色为新增的 `weak_recovered`，品红色为
`weak_bootstrap`。每次 recovery-only 都从 `original_graph.p` 开始，不会累加上一次恢复结果。

本目录是独立开发测试工具，不接入正式 GUI、`user_pipeline.py` 或正式用户工作流。

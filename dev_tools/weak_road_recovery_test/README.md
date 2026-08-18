# Weak road recovery test

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

黄色为原始 SAMRoad 中心线，青色为新增的 `weak_recovered` 部分。每次 recovery-only 都从 `original_graph.p` 开始，不会累加上一次恢复结果。

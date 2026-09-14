# Radiometric normalization A/B

本目录是独立实验，原图、正式项目缓存、正在进行的 Fast2 实验均不写入。

- 真实数据：验证区 `20250118 → 20260203`；reference 固定为 `20250118`。
- A：原始 TIFF 的逐字节副本；B：reference 的逐字节副本和后期的独立归一化 TIFF。
- 归一化：验证区有效像元每 8 像元抽样；每通道中位数匹配，IQR 估计增益；用 p1/p99 限制增益，使主要有效像素不越过 0–254。255 是原图 NoData，逐通道保持。无 GT、CLAHE、均衡或几何重采样。
- 两组都从空的实验缓存运行 SAMRoad → Fast surface → SAM-MoLRA/测宽 → 产品导出。模型配置及 Fast2 阈值相同。Fast2 的背景宽度校正保留不变。
- 不向推理和 Auto 传 GT；`evaluate.py` 必须确认两组完成且没有 GT 输入，冻结预测文件哈希后才复制和打开 GT。
- A/B 均只使用这两个时期，没有调用正式任务的相邻期成果。

## 代码冻结

本轮开始时工作树干净。运行过程中检测到其他任务修改共享 Fast2 代码，因此从开始时的提交 `734fe17cada16effc181336bfe1b18ad5e75daad` 冻结生产代码到 `snapshot/code`。
A 的提取阶段所用源代码与快照一致；B 使用快照运行全部流水线。最后 `frozen_fast2.py` 在两组未经最终协调的 Auto 道路成果上使用同一快照重新检测，报告只使用这两个 `frozen_fast2` 结果。
快照和原代码的比较记录见 `config/snapshot.json`。不修改任何生产文件。

## 文件

- `config/experiment.json`、`config/command_*.json`：固定参数和实际命令。
- `inputs/raw`、`inputs/normalized`：输入 TIFF；`normalization.json`：线性参数、分位数和保真检查。
- `normalized_tiffs`：供 GIS 查看使用的归一化 TIFF，移除了原图留下的旧统计标签；已验证像元、掩膜和网格与 B 实际输入完全一致。实际运行输入保持原样用于复现。
- `A`、`B`：各自的缓存、完整流水线产品、独立 Fast2 输出。
- `logs`、`tmp`、`cache`：实验运行日志和运行时缓存。
- `diagnostics/radiometric_inputs.png`：统一 0–255 显示范围的 RGB 预览。
- `evaluation`：GT 后验评价、无 GT 交集对象、固定采样点及整段宽差表。
- `metrics.json`：机器可读结果；`REPORT.md`：最终结论。

## 执行

在仓库根目录使用 `runtime/env/samroad_env/python.exe -B` 执行本目录脚本。已完成的目录不要再次 prepare 或 normalize。

```powershell
runtime/env/samroad_env/python.exe -B experiments/radiometric_normalization_ab/run_experiment.py A
runtime/env/samroad_env/python.exe -B experiments/radiometric_normalization_ab/run_experiment.py B
runtime/env/samroad_env/python.exe -B experiments/radiometric_normalization_ab/frozen_fast2.py
runtime/env/samroad_env/python.exe -B experiments/radiometric_normalization_ab/evaluate.py
runtime/env/samroad_env/python.exe -B experiments/radiometric_normalization_ab/test_metrics.py
```

seed=0；关闭 cuDNN benchmark 并启用 deterministic 选择。GPU 算子仍可能存在数值非确定性，reference A/B 的重复提取结果用于检查这一点。所有源模型只读。

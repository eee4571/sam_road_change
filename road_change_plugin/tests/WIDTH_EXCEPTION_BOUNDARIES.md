# 道路局部未知异常：保护边界与第二轮测试报告

2026-10-12，基于主分支 `a954bd9` 的 `fast + raw_image` 实际调用链。此次沿用已有宽度算法、回退顺序、检查点与质量分级，没有新增运行模式。

## 哪些位置可以恢复

只有输入几何已验证、单道路结果可丢弃/撤销、共享资源操作有明确边界的计算，才启用最终 `Exception` 保护；随后仍要通过系统异常检查。

| 文件 | 本次修改 |
| --- | --- |
| `width/width_reliability.py` | 统一检查系统异常及异常原因链；保留原 `ROAD_ERRORS` 给更大范围的旧边界使用，未全局扩大白名单。诊断增加 `fallback_reason`。 |
| `width/raw_image_backend.py` | 道路采样、剖面转换、重建、属性/宽度汇总、插值、宽度段转换、代表宽度的最终保护。给可变剖面传独立副本；验证剖面所属道路、必需字段、站点及坐标。异常道路最终为 C。 |
| `width/raw_boundary_core.py` | 把共享 RGB 读取/特征缓存/投影的未知异常标为系统边界失败，不能被外层误判为道路错误。 |
| `width/raw_road_surfaces.py`、`width/road_pair_matcher.py` | 道路面剖面和 Buffer/属性异常隔离；必要时保守 Buffer。如果同一道路连保守面也无法生成，保留轴线并记录缺失面。 |
| `canonical_road_surface.py`、`formal_road_products.py`、`geometry_recovery.py` | 在确认局部的宽度属性、剖面、稳定化、门户和道路面构建事务中启用未知异常保护，回滚局部写入，保留有效轴线并传播 C 级。共享路网构造/整批合并未增加兜底。 |
| `fast_pipeline.py`、`road_network_products.py` | 单道路切片导出保护，串联质量降级、异常详情和区域质量统计。正常道路记录及公开字段保持兼容。 |

`raw_width_reconstruction.py` 的正常计算公式和求解器未修改；保护位于调用它的单道路事务边界。随后按新增内存要求，将最近锚点的平方矩阵替换为等价的排序查找；详见 [RGB_MEMORY_SAFETY.md](RGB_MEMORY_SAFETY.md)。

## 必须中止的边界

- `MemoryError`、原生内存分配/资源耗尽；OpenCV 内存错误码 `-4`。
- `OSError` 及其子类（含权限、磁盘/文件读取和写入），缺失依赖、环境错误。
- `InputContractError`，无法解释的 CRS、影像网格或必要输入契约。
- 共享读取/特征缓存内发生的未知错误，即使原始类型是 `KeyError`，也中止。
- 取消、`KeyboardInterrupt`、`SystemExit`；未能证明局部性的 `RuntimeError`、`SystemError`、断言/不变量失败。
- 单道路计算之外的异常，如检查点序列化、`future.result()` 外泄错误、整批合并、文件导出。
- Python 无法捕获的原生进程崩溃/操作系统强制终止。此前已原子提交的检查点仍可供续跑，当前未完成对象重新计算。

检查异常的 `__cause__` 和未屏蔽的 `__context__`，避免把包装成 `TypeError`/`ValueError` 的系统失败当成局部成功。没有 `except Exception: pass` 或强制返回零的逻辑。

## 结果与状态

异常道路优先沿用原始可靠断面和已有多级回退，最终强制 C 级，不参与确定的自动宽变。
诊断保留道路/切片 ID、阶段、完整 traceback、回退动作和原因；正式输出沿用 `width_source`、`outlier_reason`/`qa_reason` 等字段。
GUI、CLI、完成标记和结果文件类型没有改变。警告完成仍可复用；真正的系统错误不会写成功完成标记。

## 可执行测试

```powershell
runtime/env/samroad_env/python.exe tests/test_width_local_exceptions.py -v
runtime/env/samroad_env/python.exe tests/test_width_production.py -v
runtime/env/samroad_env/python.exe tests/check_width_resilience.py -v
runtime/env/samroad_env/python.exe tests/check_formal_period_products.py -v
runtime/env/samroad_env/python.exe tests/check_fast2_structure.py -v
runtime/env/samroad_env/python.exe tests/check_processing_optimizations.py -v
gui_env/python.exe tests/test_width_warning_ui.py -v
gui_env/python.exe -m unittest discover -s tests -p test_ui_processing.py -v
```

本组结果：**168 个测试通过**（15 + 33 + 17 + 17 + 23 + 25 + 3 + 35）。另新增 11 个 RGB 内存测试，合并验收共 **179 个测试通过**，见 [RGB_MEMORY_SAFETY.md](RGB_MEMORY_SAFETY.md)。
新增 15 个测试中包含 48 组“6 个阶段 × 8 类局部异常”注入，类型包括 KeyError、TypeError、AttributeError、ValueError、IndexError、Shapely、OpenCV 和自定义异常。
另覆盖正式成果宽度/道路面故障、非法宽度属性、共享缓存损坏、系统错误包装、取消、权限错误、序列化错误、损坏后的检查点续跑。
区域集成注入同时检查质量报告、GPKG、警告完成缓存复用和健康道路质量不变；修复了混合回退记录带来的空 WKB 字段污染。
新子进程测试确认局部 KeyError 退出码为 0 且有警告报告，PermissionError 退出码非零。
对照断言检查另一条正常道路的总宽、左右距离、质量值逐项不变；回退道路的轴线集合和道路/切片关联保持正确。

缓存项目回归：`tests/check_existing_period.py ../project/plugtest` 通过。区域 1、期次 20221020、600 米子集，44 条输入中心线、55 条恢复线、4 条测宽链、141 个断面。
本轮测宽约 2.99 秒（与其他回归并行，不用作性能比较），输出 `project/plugtest/_work/_processing_check/period-20221020-8cb80586/`。
未重跑模型推理，未覆盖正式项目成果。

未验证故障电脑的原始数据、完整长时序和历史 full 模式；仍可能因上述系统/共享状态错误中止，这是刻意保留的失败边界。

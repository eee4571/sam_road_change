# Fast 正式成果链

所有生产入口调用 `code/user_pipeline.py`，不需要运行 tests 或 review 目录中的脚本。算法模块本轮没有修改。

## 最终调用链

单期 Auto 提取 → `_run_fast_change_result(..., defer_finalization=True)` 完成 Auto 变化和可选 GT 后验局部校正 → `_finalize_fast_manifest` → `build_temporal_outputs` → 现有 `engine.fast_gt_reconciliation.build_fast_temporal_outputs` 完成 Final Period Roads、Final Changes 和一次 Final Temporal → 对 Final Changes 执行 `_evaluate_existing_changes_impl` → 汇总评价 → ResultPublisher 正式发布。

有 GT 和无 GT 共用该收尾接口。无 GT 的评价状态为 `skipped_no_truth`，旧指标引用和文件不会继续作为有效评价。评价使用原始 GT，不评价中间 Auto 图层；Auto 缓存继续留在内部用于续跑。

## 正式入口

| 入口 | Fast 行为 |
|---|---|
| `run_all` / `all` | 所有变化对准备后统一收尾、评价、发布；中途只保存内部断点 |
| `change-project-periods` | 单变化对准备后调用统一收尾，发布最终期次、变化、长时序及评价 |
| `change`（Fast 期次输入） | 同一收尾接口，返回最终成果路径和关联期次、长时序 |
| `rerun-period` | 重跑该期，更新受影响相邻变化对，再统一收尾和评价 |
| `rerun-change` | 复用内部 Auto 单期输入，替换该变化对的校正，再统一协调、收尾和评价 |
| `rerun-all-periods` | 先完成期次重跑，再完成相邻变化对，最后收尾一次 |
| `rerun-all-changes` | 复用单期 Auto 缓存，完成所有变化对后收尾一次 |

Fast 的 `--update-related`、`--update-temporal` 旧参数仍可接受，但不再决定是否跳过必要下游；Full 模式保留原行为。CLI 参数和 GUI/TaskManager 调用签名保持不变。

重新协调可能影响共享期次的其他变化对，因此每次收尾后重新评价当前有 GT 的最终变化对，避免保留与新几何不对应的指标。

## 发布和续跑

开始重跑时，内部 manifest 将 Fast 收尾状态设为 `pending`，清除旧 `final_period_results`、`temporal_results` 和评价引用。该状态不能发布临时成果，也不能进入 GUI 的 manifest 回退索引。已有正式发布快照保持可用，直到新的完整收尾成功。

期次重跑从 `auto_period_results` 定位输入工作区，避免在上轮 Final 目录内做提取。变化重跑不继承旧 `correction_audit`，包括移除 GT 后转为无 GT 的情况。旧任务没有完成标记时，续跑需要经过正式收尾；新任务只有全部收尾和评价成功后才记录完成标记。

正式主目录：

- `01_单期道路/`
- `02_变化检测/`，包含 `精度评价/` 汇总
- `03_长时序/`

`road_state.gpkg`、correction/perturbation audit、Auto 缓存、source/QA/truth metadata 留在内部任务目录。发布字段使用业务白名单。旧 Auto、GT-assisted、review 和 Fast 原 `04_精度评价` 目录迁入内部发布历史；不会创建第二套 temporal。

## 手动检查

1. 无 GT 的完整 Fast 运行：三个正式目录、一次 temporal、评价正常跳过。
2. 有 GT 或部分变化对有 GT 的完整运行：中间结果不发布；最终评价与 Final Changes 对应。
3. 三期以上项目重跑中间一期：只重新检测前后相邻变化对，随后统一协调期次和长时序、刷新评价。
4. 重跑一个变化对，包括取消旧 GT：不残留旧校正或旧评价；其他 Auto 缓存不被改写。
5. 两种批量重跑及失败后续跑：最终收尾只执行一次，失败的中间结果不覆盖正式快照。
6. 已完成任务续跑、旧版本任务续跑：新 Final 可复用；缺少完成标记的旧任务需要补齐收尾。
7. GUI 打开各类成果：指向当前 Final；正式属性表不含来源/QA，单期目录没有 road_state 或 audit。

本轮只执行临时目录中的轻量回归，未运行真实项目、遥感模型推理或真实区域效果验收。

# Fast GT-assisted 依赖链

Fast 的 GT-assisted 在独立 Auto 完成后执行。GT 不参与 Auto 候选生成、precision qualification 或网络组装，也不触发用 GT 修路后重跑 Auto。

```text
原始 T1 / T2 Auto Final Roads
  → 独立 Auto Changes（保存）
  → GT 规则扰动与变化修正
  → 独立 GT-assisted Changes
  → GT-assisted Final Centerline / Width / Corridor / Road State
  → GT-assisted Temporal

原始 Auto Roads + Auto Changes → 独立 Auto Temporal
```

## 变化修正与道路协调

`engine/fast_gt_reconciliation.py` 负责完整后续链。GT 采用固定种子 4571，按对象几何和期次派生随机状态：整对象遗漏概率 2%，外端最多损失长度的 1% 且不超过 2 m，平滑横移不超过 0.4 m，宽度扰动不超过 3%，类型误差概率 0.3%。这些是概率上限配置，不保证小样本必然出现遗漏或类型误差。路口连接端点固定；不引入随机断裂、像素噪声或孔洞。

GT 面仅用于提取道路轴和宽度依据，最终面由规则轴与宽度重建。Auto 中没有明确冲突的变化区间保留；GT 补足未覆盖区间，明确冲突的类别仅替换对应纵向区间。原 Auto 文件不覆盖。

道路修改使用轴方向、投影范围、横向偏移稳定性和 track 歧义检查，禁止按 GT 面相交范围直接删路。缺失目标车道时，另一期仍有稳定对应的相邻车道受到保护。Added 协调为 absent/present，Removed 为 present/absent，宽度变化为 present/present。每一期同步写中心线、分段宽度、规则走廊和显式道路状态。

多相邻期变化先按共享纵向边界切分为稳定 track，再协调共享期次。共享宽度一致化后回写两侧变化；存在状态冲突或无法满足的宽差符号冲突会显式报错，避免输出互相矛盾的成果。只有变化类型、没有前后宽度真值时，使用原期次宽度关系推定并记录 `gt_class_with_period_width_relation`，不能视为实测宽度。

## 长时序与独立发布

Fast 完整任务、局部重跑、下游刷新及独立变化对命令均接入 temporal analysis，不再返回 `skipped_fast_profile`。GT track 的 observation 先核对实际期次中心线存在性与宽度，再生成 lifecycle、event 和 width evolution；普通道路继续使用原匹配与 review 机制。

成果发布保留原 Auto 期次目录，并增加以下变体：

- `01_单期道路/<期次>/GT-assisted/`：独立协调期次成果及 `road_state.gpkg`。
- `02_变化检测/<变化对>/Auto/` 和 `GT-assisted/`：独立变化。
- `03_长时序/Auto/` 和 `GT-assisted/`：两套长时序，均包含 `width_evolution.csv`。

CLI 参数不变；结果索引增补变体、期次状态及宽度演化路径。`user_pipeline.py` 负责调用顺序，`app/result_publisher.py` 负责独立发布。道路提取、原 Final 产品生成、Auto 判断与组装模块保持不变。

## 验证与复现

相关回归覆盖 Added / Removed / Widened / Narrowed、类型冲突、固定种子、端点连续、邻近车道保护、三期局部灭失和共享期宽度一致性。真实区域验收复用已经完成的 Auto 缓存，不重新推理：

```powershell
runtime/env/samroad_env/python.exe code/tests/run_fast_gt_chain_review.py project/test_area/auto_geometry_20260906_release 'C:/Users/zhoum/geesing/验证区/数据2（20221020）/道路.shp' project/test_area/gt_assisted_chain_NEW
```

输出目录须为新目录。脚本检查源文件哈希、期次几何、逐变化宽度/走廊、road state / temporal event，并发布局部对照图与 README。2026-09-06 验收见 `project/test_area/gt_assisted_chain_20260906_final/README.md`。该真值只有 5 个 Added 对象；其他类别的 GT 补充及冲突修正由构造回归用例验证，不能声称已获得真实区域其他类别 GT 验证。

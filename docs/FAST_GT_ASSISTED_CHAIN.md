# Fast 单套最终成果链

```text
单期道路识别 → 独立 Auto 变化检测 → 可选 GT 后验校正
→ 更新 GT 涉及的 Final Period Roads → Final Changes → 一次 Final Temporal
```

无 GT 时直接以 Auto 道路和变化作为最终成果。有 GT 时只修改 GT 对应道路轴、方向、track 和纵向区间；未涉及的原 Auto 道路和变化面保留，绝不按 GT polygon overlap 删除邻路。Auto 检测不接收 GT，也不使用校正道路重新运行 Auto。

## 重建与一致性

`engine/fast_gt_reconciliation.py` 负责后验校正和期次协调，`road_network_products.rebuild_corrected_road_axes` 复用现有 Final Road 中的低频轨迹平滑、道路端部附着和路口约束。使用米制局部坐标保留弯路，GT 只提供路口连接的支持证据，不裁切最终边界。

宽度扰动按 GT 对象共享，避免分支之间随机跳宽。单期面与变化面共同使用 `continuous_road_geometry`：先按端点拓扑连接相容轴段，再平滑跨段宽度并生成连续偏移边界。内部转弯使用偏移弧线，交汇道路消除内部边界，只有真正外端使用平直端帽。原中心线和变化判定保留，正式变化几何统一重建；原区间属性保留在内部审计中供长时序关联。

原 Road Surface 是检测证据，不能直接混入正式路面。无论是否经过 GT 校正，正式 Road Surface / Corridor 都由最终中心线及平滑 Final Width 生成，并融合路口内部边界；因此原 Auto 识别和宽度数据不变，但其正式路面也统一为规则表达，不再沿用像素化边缘。

路口附着的极短终端支段（长度不超过 1.25 倍路宽，且连接至成熟长道路）不单独生成矩形补丁，沿主路方向保留其纵向投影范围，以连续方头表达；内部连接段和孤立短道路保留。这只改变正式面表达，原始轴区间及事件关联仍保留在审计记录。局部面与旧逐段 buffer 的面积差异单独报告，不能宣称原逐段面全部原样覆盖。

Added 为 absent/present，Removed 为 present/absent，Width Change 为 present/present；中心线、宽度、走廊和内部状态表同步更新。多相邻期用共享纵向区间维护 track；共享期宽度统一后回写变化。标签冲突按连续道路支持自动选择期次状态、调整相邻变化类别，低置信度和冲突决策记录到内部审计。

## 自动匹配与 QA

跨期匹配在方向、纵向范围和连续性约束内，按综合得分选择最高候选并确定性打破平分。低置信度不会导致人工 review 或阻止正式输出。单期疑似漏检保留实际观察到的状态，仅写 QA。

可靠状态事件不依赖 GT 或变化 polygon 的存在。已有 Final Changes 正常进入事件表；若完整道路 observation 与局部变化分类不同，优先匹配方向和范围合理、类别一致的 road；仍有冲突时保存实际观察状态和正式变化类别，在内部记录 `pair_class_observation_disagreement`。这不会伪造期次道路存在性，也不代表此类事件获得人工精度认证。

## 发布与缓存

正式目录只有：

- `01_单期道路/<期次>/`
- `02_变化检测/<变化对>/`
- `03_长时序/`

不发布 Auto / GT-assisted 子目录、第二套 temporal 或人工 review 图层。正式属性使用业务字段白名单，移除 GT、Auto、source、QA、truth、review 等来源和审核字段；内部缓存保留完整信息。发现旧的双套子目录时，先发布最终成果，再将旧目录移到 `_work/tasks/publication_history`。

批量和变化对重跑先准备各对校正，最后统一协调并只构建一次区域长时序。独立变化对命令完成同样顺序。原 Auto 缓存在 `auto_period_results`、变化内部 `automatic` 中供续跑和调试；发布仅读取当前 `final_period_results`、`change_results` 和 `temporal_results`。

CLI 参数不变。原 GT 整对象遗漏、端部损失、低频偏移、微小宽度和极低概率类型扰动保留固定种子；没有碎片、孔洞或随机断裂扰动。

## 真实区域验收

```powershell
runtime/env/samroad_env/python.exe code/tests/run_fast_final_chain.py project/test_area/fast_final_NEW
```

输出目录必须是新目录。脚本复用 `auto_geometry_20260906_release`，检查原缓存哈希、Auto 判定保留和中心线、GT 状态/事件一致性、单次 temporal、正式属性和目录，并输出局部图。真实 GT 只有 5 个 Added 对象，Removed / Width GT 修正由构造回归测试覆盖。

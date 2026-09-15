"""One 4M iteration-sample run; reuse Full, compare normalized TIFF pixels only."""
import os
from run_experiment import ROOT, environment, save, sha
os.environ.update(environment())
import time
import numpy as np
import rasterio
from irmad_rrn import OUT, read, run, windows

FAST=OUT/'fast_4m'


def compare_pixels(full_path,fast_path):
    count=np.zeros(3,dtype=np.int64)
    absolute=np.zeros(3);squared=np.zeros(3)
    with rasterio.open(full_path) as a,rasterio.open(fast_path) as b:
        assert (a.crs,a.transform,a.shape,a.nodata,a.dtypes)==(b.crs,b.transform,b.shape,b.nodata,b.dtypes)
        for window in windows(a):
            mask=a.read_masks(window=window)>0
            assert np.array_equal(mask,b.read_masks(window=window)>0)
            x=a.read(window=window);y=b.read(window=window)
            assert np.array_equal(x[~mask],y[~mask])
            delta=y.astype('float64')-x
            count+=mask.sum(axis=(1,2))
            absolute+=(abs(delta)*mask).sum(axis=(1,2))
            squared+=(delta*delta*mask).sum(axis=(1,2))
    return dict(per_band_valid_pixels=count.tolist(),mae_dn=(absolute/count).tolist(),
                rmse_dn=np.sqrt(squared/count).tolist(),overall_mae_dn=float(absolute.sum()/count.sum()),
                overall_rmse_dn=float(np.sqrt(squared.sum()/count.sum())),
                nodata_mask_and_grid_equal=True)


def main():
    if FAST.exists():raise FileExistsError('Fast 4M already exists; do not rerun')
    full=read(OUT/'normalization.json')
    full_files=[OUT/'normalization.json',OUT/'paired_valid_rgb.bin']+[__import__('pathlib').Path(o['output']) for o in full['outputs']]
    identity={str(p):sha(p) for p in full_files}
    FAST.mkdir()
    save(FAST/'full_identity_before.json',identity)
    run(output_dir=FAST,fit_sample_limit=4_000_000)
    fast=read(FAST/'normalization.json')
    tick=time.perf_counter()
    native=compare_pixels(OUT/'normalized_native/20260203.tif',FAST/'normalized_native/20260203.tif')
    # The pixel-pair cache must be byte-identical: sampling is the only algorithm change.
    assert sha(FAST/'paired_valid_rgb.bin')==identity[str(OUT/'paired_valid_rgb.bin')]
    assert identity=={str(p):sha(p) for p in full_files}
    result=dict(full_reused=True,full_files_unchanged=True,all_valid_pixel_cache_identical=True,
        full_iteration_pixels=full['pixels'],fast_iteration_pixels=fast['iteration_pixels'],
        native_normalized_T2_comparison=native,comparison_seconds=time.perf_counter()-tick,
        full_timings=full['timings'],fast_timings=fast['timings'],
        gain_delta=(np.array(fast['gain'])-full['gain']).tolist(),
        offset_delta=(np.array(fast['offset'])-full['offset']).tolist(),
        rho_delta=(np.array(fast['model']['rho'])-full['model']['rho']).tolist(),
        pif_fraction_delta=fast['pif_fraction']-full['pif_fraction'])
    save(FAST/'comparison.json',result)
    write_report(full, fast, result)
    print('FAST COMPARISON COMPLETE',native,flush=True)


def write_report(full, fast, result):
    """Render saved measurements without repeating normalization or comparison."""
    native=result['native_normalized_T2_comparison']
    rows=[]
    for i,name in enumerate(('R','G','B')):
        rows.append(f"| {name} | {full['gain'][i]:.8f} / {fast['gain'][i]:.8f} | {full['offset'][i]:.6f} / {fast['offset'][i]:.6f} | {native['mae_dn'][i]:.6f} | {native['rmse_dn'][i]:.6f} |")
    total_full=full['timings']['normalization_elapsed_including_export_fix_seconds']
    total_fast=fast['timings']['normalization_total_seconds']
    conclusion='本对数据上，Fast 的最终归一化影像与 Full 很接近，支持将其作为后续流程的加速候选；尚不能视为已经完成正式流程验证。'
    text=f'''# Fast IR-MAD（最多 400 万迭代像元）对比

**{conclusion}** 本轮未修改正式代码，未运行道路提取或变化检测；只确认归一化数值近似，不代表验证了道路精度或其他影像对。

- 固定 20250118 为 reference，归一化 20260203；Full 直接复用。
- Full 迭代 {full['pixels']:,} 像元；Fast 从同一有效像元序列等分区间、确定性取中点，取 {fast['iteration_pixels']:,} 像元，无随机种子或语义筛选。
- 只有迭代拟合抽样；最终 NCP、PIF >0.95 和 TLS 仍在全部 {fast['pixels']:,} 个有效像元上计算，PIF 图和全部 normalized TIFF 的写出、nodata/网格/元数据处理沿用同一函数。均最多 30 次，rho 最大变化 <0.01 收敛。

| 时间 / 模型 | Full（已有） | Fast |
|---|---:|---:|
| 总耗时 s | {total_full:.2f} | {total_fast:.2f} |
| IR-MAD 拟合阶段 s | {full['timings']['irmad_fit_seconds']:.2f} | {fast['timings']['irmad_fit_seconds']:.2f} |
| 收敛迭代次数 | {full['selected_iteration']} | {fast['selected_iteration']} |
| 全量 PIF 像元数 | {full['pif_pixels']:,} | {fast['pif_pixels']:,} |
| 全量 PIF 比例 | {full['pif_fraction']*100:.6f}% | {fast['pif_fraction']*100:.6f}% |

**耗时口径限制：Full 的 {total_full:.2f}s 历史记录包含当时 TIFF 导出失败后的修复等待和重写，不是干净的算法基准；因此不能把两者总耗时比当作严格加速倍数。** Fast 总耗时包含重新构建有效像元缓存、抽样迭代、全量 PIF/TLS、PIF 图、8 个分析瓦片和原生 T2 TIFF 写出及保真验证，不含本次 MAE/RMSE 和哈希比较。可比性更强的拟合阶段（Fast 含抽样）从 {full['timings']['irmad_fit_seconds']:.2f}s 降至 {fast['timings']['irmad_fit_seconds']:.2f}s，约 {full['timings']['irmad_fit_seconds']/fast['timings']['irmad_fit_seconds']:.1f} 倍。未为补齐时间基准而重跑 Full。

| 波段 | gain（Full / Fast） | offset（Full / Fast） | MAE DN | RMSE DN |
|---|---|---|---:|---:|
{chr(10).join(rows)}

rho 为三个典型相关分量（不是 RGB 波段）：
- Full：{np.round(full['model']['rho'],9).tolist()}
- Fast：{np.round(fast['model']['rho'],9).tolist()}
- 最大绝对 rho 差：{max(abs(np.array(result['rho_delta']))):.9f}。

两张原生网格 normalized T2 在每波段有效像元上逐像素比较，排除 nodata：整体 **MAE={native['overall_mae_dn']:.6f} DN、RMSE={native['overall_rmse_dn']:.6f} DN**（uint8 0–255 编码）。两者网格、逐波段掩膜和无效像元一致。Full 参数、缓存和 normalized TIFF 的哈希未变。

gain 最大相对差为 {max(abs(np.array(result['gain_delta'])/full['gain']))*100:.3f}%，offset 最大绝对差为 {max(abs(np.array(result['offset_delta']))):.3f} DN。输出 RMSE 小于 1 DN，说明本对数据的最终 uint8 影像接近；但 PIF 数量相对增加 {(fast['pif_pixels']/full['pif_pixels']-1)*100:.2f}%，rho 也有可见差异，不能宣称模型或 PIF 等价。Fast 在同一收敛准则下提前一轮结束，这可能贡献差异，本轮不另跑实验归因。影像误差是在量化、裁剪后的输出上测量，不代表未裁剪浮点值完全一致。

建议：本对数据可采用 Fast 作为后续验证的归一化输入；是否成为正式默认仍需后续业务验证。本轮保留隔离实现，不改正式流程，也不追加道路或变化检测实验。

Fast 输出：`normalized_native/20260203.tif`、`normalized_tiles/`；详细记录：`normalization.json`、`comparison.json`。所有生成物继续被实验 .gitignore 排除。
'''
    (FAST/'REPORT.md').write_text(text,encoding='utf-8')


if __name__=='__main__':main()

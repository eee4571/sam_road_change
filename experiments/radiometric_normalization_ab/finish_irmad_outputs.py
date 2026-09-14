"""Resume an interrupted raster export, without refitting IR-MAD or running inference."""
import time
import numpy as np
import rasterio
from irmad_rrn import ROOT, OUT, read, paired_tiles, windows, write_normalized
from run_experiment import save
from run_irmad_experiment import identity


def main():
    cfg = read(OUT/'normalization.json')
    gain, offset = np.array(cfg['gain']), np.array(cfg['offset'])
    outputs, pifs = [], []
    for _, source in paired_tiles():
        dest = OUT/'normalized_tiles'/source.name
        clips, n = np.zeros((3,2), dtype='int64'), 0
        with rasterio.open(source) as a, rasterio.open(dest) as b:
            assert (a.crs,a.transform,a.shape,a.nodata,a.dtypes)==(b.crs,b.transform,b.shape,b.nodata,b.dtypes)
            for w in windows(a):
                values, mask = a.read(window=w), a.read_masks(window=w)>0
                assert np.array_equal(a.read_masks(window=w),b.read_masks(window=w))
                transformed = values*gain[:,None,None]+offset[:,None,None]
                expected = values.copy()
                expected[mask] = np.rint(np.clip(transformed,0,255))[mask].astype('uint8')
                assert np.array_equal(expected,b.read(window=w))
                for j in range(3):
                    clips[j] += [np.count_nonzero((transformed[j]<0)&mask[j]),np.count_nonzero((transformed[j]>255)&mask[j])]
                n += int(mask.all(0).sum())
        outputs.append(dict(source=str(source),output=str(dest),valid_pixels=n,clipped_low_high_per_band=clips.tolist(),grid_mask_nodata_verified=True))
        with rasterio.open(OUT/'pif'/source.name) as p:
            valid, count = 0, 0
            for w in windows(p):
                z = p.read(1,window=w)
                valid += int((z>=0).sum())
                count += int((z>.95).sum())
        pifs.append(dict(tile=source.name,valid=valid,pif=count))
    tick=time.perf_counter()
    source, dest = ROOT/'inputs/raw/20260203.tif', OUT/'normalized_native/20260203.tif'
    if dest.exists():
        # Preserve the failed local artifact for diagnosis; never overwrite it.
        dest.rename(dest.with_name('20260203_failed_shared_mask.tif'))
    outputs.append(write_normalized(source,dest,gain,offset))
    cfg['outputs'], cfg['pif_by_tile'] = outputs, pifs
    cfg['timings']['native_export_recovery_seconds']=time.perf_counter()-tick
    cfg['timings']['normalization_elapsed_including_export_fix_seconds']=time.time()-(OUT/'normalize.log').stat().st_ctime
    cfg['export_recovery']='Per-band nodata mask preservation fixed; existing aligned outputs verified pixel-for-pixel without rerunning normalization.'
    save(OUT/'normalization.json',cfg)
    before = read(OUT/'identity_before.json')
    assert before==identity()
    save(OUT/'normalize_audit.json',dict(baseline_and_raw_unchanged=True, checked_files=len(before), all_output_grids_masks_nodata_verified=True))
    print('OUTPUT RECOVERY VERIFIED',flush=True)


if __name__=='__main__':
    main()

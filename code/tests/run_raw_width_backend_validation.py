"""Manual cached-period integration check; not imported by production."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch
import geopandas as gpd
from engine import fast_auto_change,fast_auto_v2

def main():
    p=argparse.ArgumentParser()
    p.add_argument('manifest',type=Path);p.add_argument('products',type=Path)
    p.add_argument('--before',default='20240918');p.add_argument('--after',default='20250118')
    args=p.parse_args();manifest=json.loads(args.manifest.read_text(encoding='utf8'))
    original=manifest['auto_period_results']
    before=next(r for r in original if r['period']==args.before)
    after=dict(next(r for r in original if r['period']==args.after))
    exported=json.loads((args.products/'fast_export_cache.json').read_text(encoding='utf8'))['result']
    assert exported['width_method']=='raw_image'
    after.update(exported)
    full=gpd.read_file(after['gpkg'],layer='width_segments')
    required={'final_left_distance','final_right_distance','final_width','final_confidence','width_source','outlier_reason'}
    assert required<=set(full)
    assert full.loc[full.width_source.isin(['propagated','unresolved']),'quality_grade'].eq('C').all()
    corridors=gpd.read_file(after['gpkg'],layer='corridors')
    assert corridors.is_valid.all()
    counts={'checked':0,'eligible_AB':0,'excluded':0}
    original_filter=fast_auto_v2._reliable_width_at
    def record_filter(scene,points):
        result=original_filter(scene,points)
        counts['checked']+=len(result);counts['eligible_AB']+=int(result.sum());counts['excluded']+=int((~result).sum())
        return result
    started=time.perf_counter()
    with patch.object(fast_auto_change,'analyze_scenes',side_effect=AssertionError('Fast1 fallback')),patch.object(fast_auto_v2,'_reliable_width_at',side_effect=record_filter):
        result=fast_auto_change.detect_final_road_changes(before,after,args.products.parent/'fast2_validation',
            before_period=args.before,after_period=args.after,position_tolerance=3.,width_change_absolute=2.,
            width_change_ratio=.2,min_change_area=4.,min_change_length=None,internal_outputs=True,
            temporal_results={'previous':next(r for r in original if r['period']=='20240106'),
                              'next':next(r for r in original if r['period']=='20260203')})
    report=dict(backend='raw_image',period=args.after,width_segments=len(full),corridors=len(corridors),
        valid_corridors=bool(corridors.is_valid.all()),quality=full.quality_grade.value_counts().to_dict(),
        source=full.width_source.value_counts().to_dict(),width_filter=counts,fast2_seconds=time.perf_counter()-started,
        fast2=result,scope='Entire cached period: official export and adjacent Fast2; no model inference or GT correction')
    (args.products.parent/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

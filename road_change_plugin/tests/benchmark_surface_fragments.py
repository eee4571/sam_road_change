"""Replay stored change axes into isolated review products; no inference or edits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import geopandas as gpd
from shapely import from_wkt,union_all

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from engine.fast_gt_reconciliation import _publish_assisted_changes


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new empty output directory')
    audit=gpd.read_file(args.audit,layer='review_candidates')
    old=gpd.read_file(args.candidates).to_crs(audit.crs)
    protected=[args.audit]+[args.candidates.with_suffix(s) for s in ('.shp','.dbf','.shx','.prj')]
    def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    original=hashes()
    result=_publish_assisted_changes(audit,args.output,'before','after',require_review=True)
    new=gpd.read_file(result['review_candidates'])
    summary=dict(intervals=len(audit),old_count=len(old),new_count=len(new),
        old_sub4m2=int((old.area<4).sum()),new_sub4m2=int((new.area<4).sum()),
        old_types=old.change_typ.value_counts().to_dict(),new_types=new.change_typ.value_counts().to_dict(),
        all_valid=bool(new.is_valid.all()),formal_count=len(gpd.read_file(result['road_changes'])),coverage={})
    for kind in ('added','removed'):
        axes=union_all([from_wkt(w) for w in audit.loc[audit.change_typ.eq(kind),'axis_wkt']])
        summary['coverage'][kind]=dict(axis_length_m=axes.length)
        for name,frame in [('before',old),('after',new)]:
            surface=union_all(frame.loc[frame.change_typ.eq(kind)].geometry)
            summary['coverage'][kind][name]=axes.intersection(surface.buffer(.001)).length/max(axes.length,1e-9)
    assert summary['all_valid'] and summary['formal_count']==0
    assert original==hashes(),'Source cache changed'
    summary['source_unchanged']=True
    (args.output/'benchmark.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('audit','candidates','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())

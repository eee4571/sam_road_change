"""Prepare a small real validation area; no workbench algorithms are imported."""
import argparse
import json
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('manifest',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text(encoding='utf8'))
    spec=manifest['input_spec'];grid=next(iter(spec['grids']))
    periods=('20240918','20250118','20260203')
    truth=spec['truths']['\0'.join([grid,*periods[:2]])]['path']
    gt=gpd.read_file(truth);metric=gt.estimate_utm_crs() if gt.crs.is_geographic else gt.crs
    gt=gt.to_crs(metric);center=gt.geometry.iloc[int(gt.area.argmax())].representative_point()
    area=args.output/'validation.shp'
    gpd.GeoDataFrame({'name':['plugin_validation']},geometry=[box(center.x-256,center.y-256,center.x+256,center.y+256)],crs=metric).to_file(area)
    rows=[]
    for period in periods:
        source=Path(spec['grids'][grid][period]['path'])
        data=source.read_text(encoding='utf-8-sig')
        files=[Path(s.strip()) for s in data.splitlines() if s.strip()]
        files=[p if p.is_absolute() else source.parent/p for p in files]
        assert all(p.is_file() for p in files)
        target=args.output/f'{period}.txt'
        target.write_text('\n'.join(str(p.resolve()) for p in files)+'\n',encoding='utf8')
        rows.append(['plugin_validation',period,str(target.resolve())])
    data=dict(areas=[['plugin_validation',str(area.resolve())]],periods=rows,
              truths=[['plugin_validation',a,b,spec['truths']['\0'.join([grid,a,b])]['path']]
                      for a,b in zip(periods[:-1],periods[1:])],
              output=str((args.output/'成果输出').resolve()),profile='fast',evaluate=True,
              run_id='plugin_fixed_pipeline',truth_type_field='BHBM')
    (args.output/'task.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    print(args.output/'task.json')


if __name__=='__main__':main()

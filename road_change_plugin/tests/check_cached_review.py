"""Read-only review regression on saved project results, without model inference."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from app.change_review import ReviewSession,source_signature


def run(manifest,output):
    output.mkdir(parents=True,exist_ok=True)
    session=ReviewSession(manifest,project_root=output,output_root=output/'published')
    results=[]
    for index,pair in enumerate(session.pairs):
        opened=session.open_pair(index)
        features=opened['features']
        source=source_signature(session.source,session.inputs)
        previews=[]
        for feature in features[:1]+features[-1:]:
            preview=session.preview(feature['id'],size=256)
            assert preview['has_images'],preview['warnings']
            previews.append(dict(id=feature['id'],has_images=preview['has_images']))
        assert source_signature(session.source,session.inputs)==source
        session.assert_current()
        results.append(dict(pair=pair,count=len(features),previews=previews,warning=opened['warning']))
    (output/'review_regression.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(pairs=len(results),features=sum(r['count'] for r in results),
                         previews=sum(len(r['previews']) for r in results)),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.manifest.resolve(),args.output.resolve())

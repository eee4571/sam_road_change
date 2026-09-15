"""Build-time copy only. Packaged runtime never reads the workbench tree."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def synchronize(source, target):
    source=Path(source).resolve();target=Path(target).resolve()
    if source==target or not (source/'code/user_pipeline.py').is_file():
        raise ValueError('Expected distinct workbench source and plugin destination')
    files=[source/'code'/name for name in ('user_pipeline.py','input_catalog.py',
           'dependency_identity.py','temporal_road_analysis.py','sitecustomize.py')]
    files += [p for folder in ('code/engine','code/app') for p in (source/folder).rglob('*.py')
              if '__pycache__' not in p.parts and 'tests' not in p.parts and p.name!='editor_manager.py']
    desired={p.relative_to(source).as_posix():p for p in files}
    # Delete stale packaged modules only after checking every resolved target.
    for old in (target/'code').rglob('*.py'):
        if old.relative_to(target).as_posix() not in desired:
            if not old.resolve().is_relative_to((target/'code').resolve()):
                raise ValueError('Unexpected linked backend file')
            old.unlink()
    snapshot={}
    for relative,path in sorted(desired.items()):
        output=target/relative;output.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,output)
        snapshot[relative]=hashlib.sha256(output.read_bytes()).hexdigest()
    config=target/'runtime/config/samroad_inference.yaml'
    shutil.copy2(source/'runtime/config/samroad_inference.yaml',config)
    snapshot['runtime/config/samroad_inference.yaml']=hashlib.sha256(config.read_bytes()).hexdigest()
    (target/'resources/backend_snapshot.json').write_text(json.dumps(snapshot,indent=2),encoding='utf8')
    print(f'Copied and hashed {len(snapshot)} backend files; runtime is self-contained.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('workbench',type=Path)
    args=parser.parse_args()
    synchronize(args.workbench,Path(__file__).resolve().parents[1])

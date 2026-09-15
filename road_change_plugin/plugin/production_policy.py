"""Plugin task policy only. All algorithms live in the bundled backend."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configuration(root=ROOT):
    return json.loads((Path(root)/'resources/production_config.json').read_text(encoding='utf-8'))


def task_arguments(args, root=ROOT):
    if not args:
        return []
    settings=configuration(root)
    action=args[0]
    if action.startswith('evaluate-'):
        return list(args)
    if action not in {'all','rerun-period','rerun-change','rerun-all-periods','rerun-all-changes'}:
        raise ValueError('插件只支持当前正式流程任务')
    remove={'--execution-profile','--width-method','--fast2-compensation'}
    result=[action];i=1
    while i<len(args):
        key=str(args[i]).split('=',1)[0]
        if key in remove:
            i += 1 if '=' in str(args[i]) else 2
        elif key in {'--irmad','--no-irmad'}:
            i+=1
        else:
            result.append(str(args[i]));i+=1
    if action=='all':
        result += ['--execution-profile','fast','--irmad']
    if action in {'all','rerun-period','rerun-all-periods'}:
        result += ['--width-method','raw_image']
    result += ['--fast2-compensation',json.dumps(settings['fast2_compensation'],sort_keys=True)]
    return result


def check_existing_task(path, root=ROOT):
    """Never relabel old raw/MoLRA extraction caches as this fixed pipeline."""
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    settings=configuration(root);spec=data.get('input_spec') or {}
    irmad=spec.get('irmad') or {}
    if (data.get('execution_profile')!='fast' or spec.get('width_method')!='raw_image' or
            not irmad.get('enabled')):
        raise ValueError('历史任务的影像预处理或测宽方法与当前插件不同，请新建完整任务；不能直接复用旧道路缓存')

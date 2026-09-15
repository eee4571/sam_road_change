"""Explicit manual end-to-end driver through the plugin's actual QProcess.

Not a unittest: this script runs real models and needs deliberate invocation.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PySide6.QtWidgets import QApplication
from plugin import create_plugin


def main():
    parser=argparse.ArgumentParser();parser.add_argument('task',type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();data=json.loads(args.task.read_text(encoding='utf8'))
    if args.resume:data['resume']=True
    app=QApplication.instance() or QApplication([])
    plugin=create_plugin();widget=plugin.create_widget()
    result={};started=time.perf_counter()
    log=args.task.parent/('resume.log' if args.resume else 'pipeline.log')
    with log.open('w',encoding='utf8',buffering=1) as stream:
        def on_log(event):
            message=event.get('message','');stream.write(message+'\n')
        def complete(event):
            result.update(event);app.quit()
        plugin.task_log.connect(on_log);plugin.task_finished.connect(complete);plugin.task_failed.connect(complete)
        plugin._controller.run('all',data)
        if not result:app.exec()
    result['seconds']=time.perf_counter()-started
    (args.task.parent/('resume_result.json' if args.resume else 'run_result.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    widget.close();plugin.shutdown()
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result.get('status')=='completed' else 1


if __name__=='__main__':raise SystemExit(main())

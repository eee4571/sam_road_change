"""Structured warnings are successful products, never suppressed failures."""
import os
from pathlib import Path
import sys
import unittest
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QProcess
from plugin.runner import Runner,PREFIX
from plugin.ui.run_log import log_entry,RunLogPanel
import json
APP=QApplication.instance() or QApplication([])


class WidthWarningUiTests(unittest.TestCase):
    def test_warning_keeps_success_and_details(self):
        runner=Runner();runner._terminal=False
        results=[];failures=[]
        runner.task_finished.connect(results.append);runner.task_failed.connect(failures.append)
        event=dict(kind='warning',status='completed_with_warnings',stage='测宽',
                   message='测宽完成，部分道路使用估计宽度；异常 1',details='quality.json')
        runner.consume_line(PREFIX+json.dumps(event))
        runner._finished(0,QProcess.ExitStatus.NormalExit)
        self.assertEqual(results[0]['status'],'completed')
        self.assertEqual(results[0]['width_warnings'][0],event)
        self.assertEqual(failures,[])
        panel=RunLogPanel(lambda:None);panel.append(dict(event=event))
        self.assertIn('quality.json',panel.width_reports)
        self.assertEqual(panel.warnings,1)

    def test_warning_does_not_override_nonzero_exit(self):
        runner=Runner();runner._terminal=False;errors=[]
        runner.task_failed.connect(errors.append)
        runner.consume_line(PREFIX+json.dumps(dict(kind='warning',status='completed_with_warnings')))
        runner._finished(1,QProcess.ExitStatus.NormalExit)
        self.assertEqual(errors[0]['status'],'failed')

    def test_road_traceback_is_displayed_as_warning(self):
        self.assertEqual(log_entry(dict(message='[Width warning] ValueError: bad road'))[0],'WARNING')


if __name__=='__main__':unittest.main()

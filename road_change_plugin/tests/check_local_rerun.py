"""Synthetic backend orchestration tests. No raster/model operations."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import user_pipeline as pipeline
from app.local_rerun import run_selection, selection_plan


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'pipeline_result.json'
        self.names=['2020','2021','2022','2023','2024','2025']
        self.manifest=dict(execution_profile='fast',job_root=self.tmp.name,
            period_orders={g:{'period_order':self.names} for g in ('north','south')},
            period_results=[dict(grid=g,period=p,result='existing') for g in ('north','south') for p in self.names],
            auto_period_results=[dict(grid=g,period=p,result='auto') for g in ('north','south') for p in self.names],
            change_results=[dict(grid=g,before_period=a,after_period=b) for g in ('north','south') for a,b in zip(self.names,self.names[1:])],
            temporal_results=[dict(grid=g,result='unchanged') for g in ('north','south')])
        pipeline.write_json(self.path,self.manifest)
        self.calls=[];self.fail=None
        def period(m,g,p):
            self.calls.append(('period',p))
            if self.fail==('period',p):raise RuntimeError('interrupt')
        def change(m,g,a,b):
            self.calls.append(('change',a,b))
            if self.fail==('change',a,b):raise RuntimeError('interrupt')
        def downstream(m):
            self.calls.append(('downstream',))
            self.assertTrue(all(r['grid']=='north' for key in ('auto_period_results','period_results','change_results') for r in m[key]))
            if self.fail==('downstream',):raise RuntimeError('interrupt')
            m['temporal_results']=[dict(grid='north',result='new')]
            m['fast_finalization_state']='completed'
        replacements=dict(_apply_fast2_task_settings=Mock(),_rerun_period_entry=period,_rerun_change_entry=change,
            _refresh_manifest_downstream=downstream,aggregate_change_evaluations=Mock(),emit=Mock(),
            _persist_existing_pipeline=lambda m,p:pipeline.write_json(p,m))
        context=patch.multiple(pipeline,**replacements);context.start();self.addCleanup(context.stop)

    def args(self,periods=(),pairs=()):
        return SimpleNamespace(pipeline_manifest=str(self.path),grid='north',
            selection=json.dumps(dict(selected_periods=list(periods),selected_pairs=list(pairs))))

    def test_mixed_selection_deduplicates_and_orders_dependencies(self):
        args=self.args(['2021','2022','2021'],[['2021','2022'],['2024','2025']])
        plan=run_selection(args,pipeline)
        self.assertEqual(self.calls[:2],[('period','2021'),('period','2022')])
        changes=[c for c in self.calls if c[0]=='change']
        self.assertEqual(len(changes),len(set(changes)))
        self.assertEqual(changes,[('change',a,b) for a,b in plan['pairs']])
        self.assertEqual(self.calls.count(('downstream',)),1)
        saved=pipeline.read_json(self.path)
        for key in ('period_results','auto_period_results','change_results','temporal_results'):
            self.assertEqual([r for r in saved[key] if r['grid']=='south'],[r for r in self.manifest[key] if r['grid']=='south'])

    def test_pair_only_never_extracts_roads(self):
        run_selection(self.args(pairs=[['2021','2022'],['2023','2024']]),pipeline)
        self.assertEqual(self.calls,[('change','2021','2022'),('change','2023','2024'),('downstream',)])

    def test_resume_road_failure_does_not_repeat_finished_period(self):
        args=self.args(['2021','2022']);self.fail=('period','2022')
        with self.assertRaisesRegex(RuntimeError,'interrupt'):run_selection(args,pipeline)
        self.calls.clear();self.fail=None
        run_selection(args,pipeline)
        self.assertEqual(self.calls[0],('period','2022'))
        self.assertNotIn(('period','2021'),self.calls)

    def test_resume_change_failure_does_not_repeat_roads_or_completed_pairs(self):
        args=self.args(['2021']);self.fail=('change','2021','2022')
        with self.assertRaises(RuntimeError):run_selection(args,pipeline)
        self.calls.clear();self.fail=None
        run_selection(args,pipeline)
        self.assertEqual(self.calls[0],('change','2021','2022'))
        self.assertFalse(any(c[0]=='period' for c in self.calls))
        self.assertNotIn(('change','2020','2021'),self.calls)

    def test_resume_downstream_does_not_repeat_changes(self):
        args=self.args(pairs=[['2021','2022']]);self.fail=('downstream',)
        with self.assertRaises(RuntimeError):run_selection(args,pipeline)
        self.calls.clear();self.fail=None
        run_selection(args,pipeline)
        self.assertEqual(self.calls,[('downstream',)])

    def test_invalid_or_changed_plan_rejected_before_work(self):
        for args in (self.args(),self.args(['missing']),self.args(pairs=[['2020','2024']])):
            with self.assertRaises(ValueError):run_selection(args,pipeline)
        self.assertFalse(self.calls)
        run_selection(self.args(['2021']),pipeline);self.calls.clear()
        with self.assertRaises(ValueError):run_selection(self.args(['2022']),pipeline)
        self.assertFalse(self.calls)

    def test_cli_keeps_old_commands_and_accepts_selection(self):
        args=pipeline.parser().parse_args(['rerun-selection','--pipeline-manifest',str(self.path),
            '--grid','north','--selection',self.args(['2021']).selection,'--width-method','raw_image'])
        self.assertEqual(args.command,'rerun-selection')
        self.assertEqual(pipeline.parser().parse_args(['rerun-period','--pipeline-manifest',str(self.path),
            '--grid','north','--period','2021','--update-related']).period,'2021')


if __name__=='__main__':unittest.main()

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from plugin.production_policy import configuration,task_arguments,check_existing_task


class FixedPolicyTests(unittest.TestCase):
    def test_full_and_reruns_force_current_policy(self):
        for action in ('all','rerun-period','rerun-change','rerun-all-periods','rerun-all-changes'):
            args=task_arguments([action,'--execution-profile','full','--width-method=sam_molra',
                                 '--no-irmad','--fast2-compensation','normalized_input'])
            config=json.loads(args[args.index('--fast2-compensation')+1])
            self.assertFalse(config['patch_radiometric_normalization'])
            self.assertTrue(all(v for k,v in config.items() if k!='patch_radiometric_normalization'))
            self.assertNotIn('full',args);self.assertNotIn('sam_molra',args)
            if action=='all':
                self.assertIn('--irmad',args)
                self.assertEqual(args[args.index('--irmad-reference')+1],'20250118')
            self.assertEqual(args,task_arguments(args))

    def test_old_extraction_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'manifest.json'
            path.write_text(json.dumps(dict(execution_profile='fast',input_spec=dict(width_method='sam_molra'))))
            with self.assertRaises(ValueError):check_existing_task(path)
            self.assertEqual(json.loads(path.read_text())['input_spec']['width_method'],'sam_molra')
            path.write_text(json.dumps(dict(execution_profile='fast',input_spec=dict(width_method='raw_image',
                irmad=dict(enabled=True,reference_period='20250118')))))
            check_existing_task(path)

    def test_backend_and_model_paths_are_inside_plugin(self):
        from plugin.runner import Runner
        runner=Runner()
        program,args=runner.command(['all'])
        self.assertTrue(Path(program).is_relative_to(ROOT))
        self.assertTrue(Path(args[1]).is_relative_to(ROOT))
        self.assertEqual(runner.environment().value('PYTHONPATH'),str(ROOT/'code'))
        self.assertEqual(runner.environment().value('SAMROAD_MODELS_ROOT'),str(ROOT/'runtime/model'))

    def test_results_do_not_publish_internal_facets_or_auto_geopackage(self):
        from plugin.result_parser import read_results
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('surface.shp','facets.shp','auto.gpkg','change.shp','width.png'):
                (root/name).touch()
            manifest=dict(period_results=[dict(width_segments='facets.shp',
                published=dict(surfaces='surface.shp',road_width='width.png'))],
                change_results=[dict(gpkg='auto.gpkg',published=dict(changes='change.shp'))])
            path=root/'result.json';path.write_text(json.dumps(manifest))
            names={Path(record['path']).name for record in read_results(path)}
            self.assertEqual(names,{'surface.shp','change.shp','width.png'})

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.fast_settings import SETTING_DEFAULTS, compensation_values, restore_settings, settings_values
from app.project_manager import ProjectManager
from app.task_manager import TaskManager, build_pipeline_command
from engine.fast2_compensation import Fast2CompensationConfig, cache_matches
from engine.fast_multitemporal import AUTO_REVISION
import user_pipeline as pipeline


class Variable:
    def __init__(self, value): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value


class FastSettingsTests(unittest.TestCase):
    def test_project_save_restore_and_missing_settings_defaults(self):
        from gui.data_page import DataPage
        variables={k:Variable(v) for k,v in SETTING_DEFAULTS.items()}
        variables['output_root']=Variable('out')
        variables['patch_radiometric_normalization'].set('0')
        variables['irmad'].set('1');variables['irmad_reference'].set('20240106')
        page=SimpleNamespace(vars=variables,project_config={},project_root_path='',project_data_sources=[],
            project_scan_cache={},project_txt_encodings={},project_path_relocations={},project_validation_areas=[],
            project_area_periods={},project_area_truths=[],project_area_truth_field_configs={},project_candidates={},
            _store_truth_field_controls=lambda:None)
        with tempfile.TemporaryDirectory() as temp:
            page.project_root_path=temp
            payload=DataPage._project_payload(page)
            ProjectManager().save_config(temp,payload)
            stored=ProjectManager().read_config(temp)
            reopened={k:Variable('wrong') for k in variables}
            restore_settings(reopened,stored['fast_settings'])
            self.assertEqual(settings_values(reopened),settings_values(variables))
            restore_settings(reopened,{})
            self.assertEqual(settings_values(reopened),SETTING_DEFAULTS)
            self.assertEqual(compensation_values(reopened),Fast2CompensationConfig().to_dict())

    def test_all_command_and_every_rerun_builder_forward_custom_flags(self):
        config=Fast2CompensationConfig(False,True,False,True).to_dict()
        base=dict(mode='grid',source_root=str(Path.cwd()),output_root='out',checkpoint='m',config='c',
                  device='cpu',pixel_size='0',rescale='off',absolute='2',ratio='.2',tolerance='3')
        commands=[build_pipeline_command(**base,irmad=True,irmad_reference='20240106',fast2_compensation=config),
                  TaskManager.build_rerun_change('manifest','g','a','b',fast2_compensation=config),
                  TaskManager.build_rerun_period('manifest','g','a',fast2_compensation=config),
                  TaskManager.build_rerun_all_changes('manifest',fast2_compensation=config),
                  TaskManager.build_rerun_all_periods('manifest',fast2_compensation=config)]
        for command in commands:
            args=pipeline.parser().parse_args(command)
            self.assertEqual(Fast2CompensationConfig.resolve(args.fast2_compensation).to_dict(),config)
        args=pipeline.parser().parse_args(commands[0])
        self.assertTrue(args.irmad);self.assertEqual(args.irmad_reference,'20240106')

    def test_selected_pair_rerun_receives_settings_and_invalidates_identity(self):
        config=Fast2CompensationConfig(False,True,True,False)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'manifest.json'
            path.write_text(json.dumps(dict(execution_profile='fast',input_spec={})),encoding='utf8')
            args=argparse.Namespace(pipeline_manifest=str(path),grid='g',before_period='a',after_period='b',
                                    update_temporal=True,fast2_compensation=config.to_dict())
            with patch.object(pipeline,'_invalidate_fast_finalization'), \
                 patch.object(pipeline,'_invalidate_change_evaluation'), \
                 patch.object(pipeline,'_persist_existing_pipeline'), \
                 patch.object(pipeline,'_refresh_manifest_downstream'), \
                 patch.object(pipeline,'_rerun_change_entry',return_value={}) as rerun:
                pipeline.rerun_pipeline_change(args)
            manifest=rerun.call_args.args[0]
            self.assertEqual(manifest['input_spec']['fast2_compensation'],config.to_dict())
            old=dict(fast_auto_revision=AUTO_REVISION,
                     fast2_compensation_identity=Fast2CompensationConfig().cache_identity)
            self.assertFalse(cache_matches(old,config))

    def test_gui_defaults_and_no_automatic_coupling(self):
        variables={k:Variable(v) for k,v in SETTING_DEFAULTS.items()}
        expected=compensation_values(variables)
        variables['irmad'].set('1');variables['irmad_reference'].set('20240106')
        self.assertEqual(compensation_values(variables),expected)
        variables['width_temporal_bias_correction'].set('0')
        self.assertEqual(variables['irmad'].get(),'1')


if __name__ == '__main__': unittest.main()

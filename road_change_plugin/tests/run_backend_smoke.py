"""Run synthetic workbench regressions against this plugin's copied backend."""
import argparse
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('test_sources',type=Path)
    args=parser.parse_args()
    sys.path[:0]=[str(ROOT/'code'),str(args.test_sources.resolve())]
    # Pin the packages before test fixtures add their own search paths.
    import engine
    engine.__path__=[str(ROOT/'code/engine')]
    import user_pipeline
    import engine.fast_auto_v2
    import app.fast_settings
    names=['test_irmad_preprocessing','test_raw_image_width_backend','test_raw_road_surfaces',
           'test_fast_auto_v2','test_fast2_compensation','test_fast_production_finalization',
           'test_rgb_production_chain','test_fast_batch_runtime','test_user_pipeline']
    suite=unittest.defaultTestLoader.loadTestsFromNames(names)
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    for name,module in list(sys.modules.items()):
        if name=='user_pipeline' or name=='engine' or name.startswith(('engine.','app.')):
            file=getattr(module,'__file__',None)
            if file and not Path(file).resolve().is_relative_to(ROOT/'code'):
                raise AssertionError(f'Backend escaped plugin: {name}: {file}')
    print('Verified all loaded backend/app modules came from the plugin copy.')
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())

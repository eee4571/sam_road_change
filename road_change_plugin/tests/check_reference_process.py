"""Real GUI -> GIS sampling -> saved configuration; injected model, synthetic rasters."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6.QtCore import QSettings, QThreadPool
from PySide6.QtWidgets import QApplication
from plugin import create_plugin
from plugin.runner import Runner

app = QApplication.instance() or QApplication([])


def wait(widget):
    deadline = time.monotonic()+90
    while (widget.browsing or widget.open_timer.isActive()) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert not widget.browsing, 'Data check timed out'
    assert widget.checked, widget.check_note.text()


with tempfile.TemporaryDirectory() as folder:
    root = Path(folder) / '自动参考项目'
    root.mkdir()
    runner = Runner()
    environment = runner.environment()
    script = '''import sys,json
from pathlib import Path
import numpy as np,geopandas as gpd,rasterio
from shapely.geometry import box
from rasterio.transform import from_origin
root=Path(sys.argv[1]); area=root/'area.shp'
gpd.GeoDataFrame(geometry=[box(0,0,2048,2048)],crs='EPSG:3857').to_file(area)
yy,xx=np.indices((2048,2048)); rgb=np.stack([60+(xx+yy+offset)%120 for offset in (0,20,40)])
periods=[]
for year,offset in [('2020',-35),('2021',0),('2022',35)]:
 image=root/(year+'.tif'); listing=image.with_suffix('.txt')
 with rasterio.open(image,'w',driver='GTiff',height=2048,width=2048,count=3,dtype='uint8',crs='EPSG:3857',transform=from_origin(0,2048,1,1),tiled=True,blockxsize=128,blockysize=128) as ds:
  ds.write(np.clip(rgb+offset,1,254).astype('uint8'))
 listing.write_text(image.name,encoding='utf8'); periods.append([year,listing.name])
(root/'project_config.json').write_text(json.dumps(dict(project_root=str(root),validation_areas=[['A',area.name]],area_periods={'A':periods},area_truths=[],area_irmad_references={})),encoding='utf8')
'''
    subprocess.run([str(runner.python_path()), '-c', script, str(root)],
                   env={k: environment.value(k) for k in environment.keys()}, check=True)
    preferences = QSettings(str(root/'ui.ini'), QSettings.Format.IniFormat)
    with patch('plugin.widget.QSettings', return_value=preferences):
        plugin = create_plugin()
        widget = plugin.create_widget()
    real_run = subprocess.run
    def run_with_test_model(command, **kwargs):
        if 'app.reference_selection' in command:
            bootstrap = (
                "import sys; sys.path.insert(0," + repr(str(ROOT/'tests')) + "); "
                "from check_reference_selection import FakeProbe; "
                "from app import reference_selection as selection; "
                "selection.RoadModelProbe=FakeProbe; selection.main()"
            )
            command = [command[0], '-u', '-c', bootstrap]
        return real_run(command, **kwargs)
    process_patch = patch('subprocess.run', side_effect=run_with_test_model)
    process_patch.start()
    try:
        widget._load_project(root)
        wait(widget)
        config = lambda: json.loads((root/'project_config.json').read_text(encoding='utf8'))
        assert config()['area_irmad_references'] == {'A':'2022'}
        assert '已自动选择：2022' in widget.configuration.reference_note.text()
        assert widget.run_button.text() == '开始处理'
        widget._load_project(root)
        with patch.object(widget.controller, 'recommend_references', side_effect=AssertionError('saved choice must stay fixed')):
            wait(widget)
            widget.configuration.reference.setCurrentText('2020')
            widget._save_and_check()
            wait(widget)
        assert config()['area_irmad_references'] == {'A':'2020'}
        assert config()['automatic_reference_selections'] == {}
        widget.configuration.reference.setCurrentIndex(0)
        widget._save_and_check()
        wait(widget)
        assert config()['area_irmad_references'] == {'A':'2022'}
        candidates = config()['automatic_reference_selections']['A']['candidates']
        assert all(item['cached'] for item in candidates.values())
        assert not (root/'_work/current/pipeline_result.json').exists()
        print('PASS: real bounded sampling/check with injected model, saved choice, reopen, manual override, cached re-selection')
    finally:
        process_patch.stop()
        widget.close()
        plugin.shutdown()
        QThreadPool.globalInstance().waitForDone(10000)
        app.processEvents()

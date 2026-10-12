"""Allocation-free probes of known full-flow risks; NOT a safety acceptance test.

Run with the backend interpreter. Large allocation calls are intercepted before
execution. A zero exit means the audit ran, not that the pipeline is memory safe.
No model, project data, inference cache, or published result is modified.
"""
import argparse
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'code/engine/width'))
sys.path.insert(0, str(ROOT / 'code/engine/samroad'))
import geopandas as gpd
import numpy as np
from rasterio.transform import Affine
from shapely.geometry import LineString, box
import yaml


class AllocationIntercepted(BaseException):
    """Deliberately outside Exception: application recovery must not hide a probe."""


def capture_allocation(module, attribute, operation, dimensions):
    requests = []
    prevented=None

    def intercepted(*args, **kwargs):
        requests.append(dimensions(args, kwargs))
        raise AllocationIntercepted()

    with patch.object(module, attribute, intercepted):
        try:
            operation()
        except AllocationIntercepted:
            pass
        except ValueError as error:
            prevented=str(error)
    return dict(reached_allocation=bool(requests), requests=requests,preflight_rejection=prevented)


def pure_inference_functions():
    # Execute only the real shape/planning functions. Importing inferencer would
    # import Torch/model dependencies and is unnecessary for these probes.
    path = ROOT / 'code/engine/samroad/inferencer.py'
    names = {'get_full_coverage_positions', 'get_full_coverage_patch_info',
             'rescale_image_to_model_gsd'}
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(body) != len(names):
        raise RuntimeError('Inference planning functions changed; update this audit.')
    namespace = dict(args=SimpleNamespace(max_image_megapixels=25.))
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def audit():
    from engine import road_axis_cleanup, auto_change_geometry, irmad_core
    from engine.road_geometry import _RegionalRoadSeed
    import road_change_detection as evaluation

    findings = {}
    line = LineString([(0, 0), (1e9, 0)])
    road = _RegionalRoadSeed(np.asarray(line.coords), 6., (1,))
    findings['regional_axis_before_rgb'] = capture_allocation(
        road_axis_cleanup.np, 'arange', lambda: road_axis_cleanup._axis_candidate(road),
        lambda a, k: dict(elements=int(np.ceil((a[1]-a[0])/a[2])),
                          float64_bytes=int(np.ceil((a[1]-a[0])/a[2]))*8))
    findings['formal_change_stations'] = capture_allocation(
        auto_change_geometry.np, 'linspace', lambda: auto_change_geometry._stations(line),
        lambda a, k: dict(elements=int(a[2]), float64_bytes=int(a[2])*8))

    w = next(irmad_core.windows(SimpleNamespace(width=1_000_000, height=128)))
    findings['irmad_wide_input_strip'] = dict(
        shape=[int(w.height), int(w.width)], pixels=int(w.height*w.width),
        paired_rgb_uint8_bytes=int(w.height*w.width)*6,
        exposure='custom/bypassed normalization; normal plugin tiles are 4096 or smaller')

    functions = pure_inference_functions()
    config = yaml.safe_load((ROOT/'runtime/config/samroad_inference.yaml').read_text(encoding='utf-8-sig'))
    shape_costs = []
    for size in (4096, 8192):
        p = int(config['PATCH_SIZE'])
        count = len(functions['get_full_coverage_patch_info'](size, size, p, config['SAMPLE_MARGIN']))
        shape_costs.append(dict(size=size, patches=count,
            fusion_float32_bytes=3*size*size*4,
            retained_two_channel_float32_masks_bytes=count*2*p*p*4,
            note='Historical eager-cache lower bound; current implementation spills bounded batches.'))
    findings['inference_array_lower_bounds'] = dict(
        infer_batch_size=config['INFER_BATCH_SIZE'], patch_size=config['PATCH_SIZE'],
        sample_margin=config['SAMPLE_MARGIN'], cases=shape_costs)

    functions['get_inference_gsd'] = lambda *args: (10., .5)
    functions['cv2'] = SimpleNamespace(INTER_CUBIC=2, INTER_AREA=3, resize=lambda *a, **k: None)
    findings['rescale_before_target_size_check'] = capture_allocation(
        functions['cv2'], 'resize',
        lambda: functions['rescale_image_to_model_gsd'](SimpleNamespace(shape=(4096, 4096, 3)), '',
                                                       {'INFER_RESCALE_TO_MODEL_GSD': True}),
        lambda a, k: dict(shape=list(a[1]), rgb_uint8_bytes=int(a[1][0])*int(a[1][1])*3))
    findings['rescale_before_target_size_check']['enabled_in_current_config'] = bool(config['INFER_RESCALE_TO_MODEL_GSD'])

    frame = gpd.GeoDataFrame({'change_typ': ['added']}, geometry=[box(0, 0, 1e12, 10)], crs=32650)
    findings['evaluation_area_cap_bypass'] = capture_allocation(
        evaluation, 'rasterize',
        lambda: evaluation._centerline_offset_metrics(frame, frame, 5., matched_pairs=[(0, 0)]),
        lambda a, k: dict(shape=list(k['out_shape']), cells=int(k['out_shape'][0])*int(k['out_shape'][1]),
                          intended_cap=8_000_000))
    frame = gpd.GeoDataFrame({'change_typ': ['added']},
                             geometry=[box(10, 10, 80_000, 80_000)], crs=32650)
    findings['assisted_evaluation_bbox'] = capture_allocation(
        evaluation, 'rasterize',
        lambda: evaluation.evaluate_fast_assisted_centerline_metrics(
            frame, frame, truth_type_field='change_typ', image_crs=frame.crs,
            image_transform=Affine.identity(), image_shape=(80_032, 80_032)),
        lambda a, k: dict(shape=list(k['out_shape']),
                          single_uint8_mask_bytes=int(k['out_shape'][0])*int(k['out_shape'][1])))
    return dict(scope='Full production Fast + raw_image + IR-MAD pipeline',
                meaning='Diagnostic findings, not a memory-safety certification',
                model_inference_executed=False, large_allocations_executed=False,
                findings=findings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit()
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+'\n', encoding='utf-8')
    print(text)

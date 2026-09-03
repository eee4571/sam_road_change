import json
import shutil
from pathlib import Path

from engine.fast_pipeline import measure_fast_widths

period = Path(
    "project/test_area/_work/tasks/runs/run_20260828_142123/grids/验证区1/periods/20221020"
).resolve()
run = period / "runs/roads_rerun_1788415454"
target = Path(
    "C:/Users/zhoum/.codex/visualizations/2026/09/03/"
    "01a065cb-9117-7ed1-9024-8f7ccdea7396/real_20221020_connectivity_final"
)
surface_input = target / "surface_input"
surface_input.mkdir(parents=True, exist_ok=True)
for source in (run / "surfaces/masks/grid_tiles").glob("v*.*"):
    if source.is_file():
        shutil.copy2(source, surface_input / source.name)
(target / "molra_probability_cache").mkdir(parents=True, exist_ok=True)
for source in (run / "width_review/molra_probability_cache").glob("*.npy"):
    shutil.copy2(source, target / "molra_probability_cache" / source.name)

summary = measure_fast_widths(
    period / "images",
    surface_input,
    run / "inference/road_graphs/grid_tiles/mask",
    target,
)
keys = (
    "original_feature_count", "final_feature_count",
    "endpoint_count_before", "endpoint_count_after",
    "dangling_endpoint_count_before", "dangling_endpoint_count_after",
    "t_junction_count", "cross_junction_count", "y_junction_count",
    "endpoint_to_endpoint_connection_count", "endpoint_to_road_attachment_count",
    "corridor_merge_count", "axis_intersection_count",
    "connected_component_count_before", "connected_component_count_after",
    "surface_center_correction_count", "generated_connection_length_m",
    "regional_regularization_seconds", "before_after_visualization", "working_gpkg",
)
print(json.dumps({key: summary[key] for key in keys}, ensure_ascii=False, indent=2))

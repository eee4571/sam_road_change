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
    "01a065cb-9117-7ed1-9024-8f7ccdea7396/real_20221020_canonical"
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
    "original_vertex_count", "final_vertex_count",
    "generated_junction_count", "merged_road_entity_count",
    "canonical_straight_road_count", "canonical_curved_road_count",
    "mean_vertices_per_road_before", "mean_vertices_per_road_after",
    "generated_connection_length_m", "regional_regularization_seconds",
    "before_after_visualization", "working_gpkg",
)
print(json.dumps({key: summary[key] for key in keys}, ensure_ascii=False, indent=2))

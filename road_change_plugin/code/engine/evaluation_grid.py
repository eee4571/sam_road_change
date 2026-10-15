"""Resolve the actual cached image grid shared by evaluation and assistance."""
import json
from pathlib import Path


def period_evaluation_grid(entries, summary=None):
    summary = summary or {}
    stored_crs = summary.get("gt_assisted_grid_crs")
    stored_transform = summary.get("gt_assisted_grid_transform")
    stored_shape = summary.get("gt_assisted_grid_shape")
    candidates = []
    for entry in entries:
        value = entry.get('result')
        result_path = Path(value).expanduser() if value else None
        payload = json.loads(result_path.read_text(encoding="utf-8")) if result_path is not None and result_path.is_file() else {}
        for record in (entry, payload):
            value = record.get('road_probability')
            if value:
                path = Path(value).expanduser()
                if not path.is_absolute() and result_path is not None:
                    path = result_path.parent / path
                candidates.append(path.resolve())
            if record.get('run_root'):
                candidates.append(Path(record['run_root']).expanduser() / 'products' / 'road_probability.tif')
    probability_path = next((path for path in candidates if path.is_file()), None)
    if probability_path is None:
        if (
            stored_crs
            and isinstance(stored_transform, list)
            and len(stored_transform) >= 6
            and isinstance(stored_shape, list)
            and len(stored_shape) >= 2
        ):
            return stored_crs, stored_transform[:6], (
                int(stored_shape[0]), int(stored_shape[1]),
            )

        attempted = '; '.join(dict.fromkeys(str(path) for path in candidates)) or 'no raster reference in final or Auto period records'
        raise FileNotFoundError(
            f"Fast evaluation probability raster missing: {attempted}"
        )
    import rasterio
    with rasterio.open(probability_path) as dataset:
        return dataset.crs, dataset.transform, dataset.shape

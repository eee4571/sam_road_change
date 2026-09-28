"""Manifest-scoped, read-only input discovery. Never scan model or project trees."""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import CRS

import bootstrap
from engine.road_connection_evidence import molra_sources, probability_sources


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def discover(project, period, area=None):
    project = Path(project).resolve()
    manifest = project / "_work/current/pipeline_result.json"
    rows = [r for r in read_json(manifest).get("period_results", [])
            if str(r.get("period")) == period and (area is None or str(r.get("grid")) == area)]
    if len(rows) != 1:
        raise ValueError(f"Expected one current period result; found {len(rows)}. Specify --area; available: "
                         f"{[(r.get('grid'), r.get('period')) for r in read_json(manifest).get('period_results', [])]}")
    record = rows[0]
    run = Path(record["run_root"])
    products = run / "products"
    lines = Path(record["centerlines"])
    if not lines.is_file():
        raise FileNotFoundError(f"Formal centerlines missing: {lines}; extraction is never launched")
    source = Path(record["source"])
    if source.suffix.lower() == ".txt":
        images = [Path(s.strip().strip('"')) for s in source.read_text(encoding="utf-8-sig").splitlines()
                  if s.strip() and not s.lstrip().startswith("#")]
        images = [p if p.is_absolute() else source.parent / p for p in images]
    elif source.is_dir():
        images = sorted(source.glob("*.tif"))
    else:
        images = [source]
    images = [p for p in images if p.is_file()]
    tile_rows = [read_json(p) for p in sorted((run / "width_review").glob("*_summary.json"))
                 if p.name != "batch_width_summary.json"]
    probability = Path(record.get("road_probability", products / "road_probability.tif"))
    probabilities = [(probability, None)] if probability.is_file() else []
    if not probabilities:
        for tile in tile_rows:
            image = Path(tile["image"])
            probabilities += [(p, im) for p, im in probability_sources(
                image.parent, run / "inference/road_graphs/grid_tiles/mask") if p.is_file()]
        probabilities = list(dict.fromkeys(probabilities))
    molra = molra_sources(run / "width_review")
    assets = dict(manifest=manifest, centerlines=lines, surfaces=Path(record["surfaces"]),
                  regional=products / "regional_products.gpkg",
                  width_profiles=products / "width_profile_cache.json",
                  observations=products / "raw_width/width_observations.gpkg")
    return dict(project=project, period=period, area=str(record["grid"]), record=record,
                assets=assets, images=images, probabilities=probabilities, molra=molra,
                missing=[k for k, p in assets.items() if not p.is_file()] +
                ([] if images else ["original_images"]) + ([] if probabilities else ["probability"]) +
                ([] if molra else ["molra"]))


def load_lines(inputs):
    frame = gpd.read_file(inputs["assets"]["centerlines"])
    if frame.crs is None:
        raise ValueError("Centerlines have no CRS")
    source_crs = CRS(frame.crs)
    metric = source_crs if source_crs.is_projected and all(
        abs(a.unit_conversion_factor - 1) < 1e-8 for a in source_crs.axis_info[:2]) else frame.estimate_utm_crs()
    frame = frame.to_crs(metric).explode(index_parts=False).reset_index(drop=True)
    frame = frame[frame.geometry.geom_type == "LineString"].copy()
    frame["source_id"] = np.arange(len(frame))
    if "width_m" not in frame:
        raise ValueError("Formal centerlines lack cached width_m; refusing to estimate road widths")
    if not np.isfinite(frame.width_m).all() or (frame.width_m <= 0).any():
        raise ValueError("Invalid saved width_m")
    return frame


def read_roi(path, crs, roi, layer=None):
    """Read only features intersecting the ROI using a bbox in the file CRS."""
    header = gpd.read_file(path, layer=layer, rows=0)
    if header.crs is None:
        raise ValueError(f"Missing CRS: {path}")
    bounds = gpd.GeoSeries([roi], crs=crs).to_crs(header.crs).total_bounds
    return gpd.read_file(path, layer=layer, bbox=tuple(bounds)).to_crs(crs)


def fingerprint(inputs):
    paths = set(inputs["assets"].values()) | set(inputs["images"])
    for path, image in inputs["probabilities"] + inputs["molra"]:
        paths.add(path)
        if image:
            paths.add(image)
    for path in list(paths):
        if path.suffix.lower() == ".shp":
            paths.update(path.parent.glob(path.stem + ".*"))
    return {str(p): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(paths) if p.is_file()}

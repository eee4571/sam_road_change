"""Run baseline, geometry ablation, image ablations and combined ROI experiments."""
import argparse
from datetime import datetime, timezone
import hashlib
import time
from uuid import uuid4

import numpy as np
from shapely.geometry import box, Point
from shapely.strtree import STRtree

from bootstrap import ROOT
from data import discover, fingerprint, load_lines, read_json
from evidence import Evidence
from graph import build, candidates
from refine import run
from report import save_lines, save_result, write_json, candidate_sheets, ribbon_overview
from ribbon_diagnostic import diagnose


def select_roi(frame, config):
    # A deterministic density proxy proposes an ROI; industrial land use needs human review.
    edges = build(frame)
    bounds = box(*frame.total_bounds)
    proposals = candidates(edges, bounds.buffer(1), config)
    locations = [c.axis.interpolate(.5, normalized=True) for c in proposals]
    if not locations:
        locations = [line.interpolate(.5, normalized=True) for line in frame.geometry]
    tree = STRtree(locations)
    size = config["default_roi_m"]
    centers = [Point(x, y) for x in np.arange(bounds.bounds[0]+size/2, bounds.bounds[2], size/2)
               for y in np.arange(bounds.bounds[1]+size/2, bounds.bounds[3], size/2)]
    if not centers:
        centers = [bounds.centroid]
    best = max(centers, key=lambda p: len(tree.query(box(p.x-size/2, p.y-size/2, p.x+size/2, p.y+size/2))))
    return box(best.x-size/2, best.y-size/2, best.x+size/2, best.y+size/2)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True)
    p.add_argument("--period", required=True)
    p.add_argument("--area", help="Validation area/grid ID; required when period is ambiguous")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    group.add_argument("--center", type=float, nargs=2, metavar=("X", "Y"))
    p.add_argument("--radius", type=float, help="Square ROI half-side in metres, used with --center")
    p.add_argument("--config", default=str(ROOT / "config.json"))
    p.add_argument("--inspect", action="store_true", help="Print cache availability and metric CRS without running")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if ((args.radius is not None) != (args.center is not None) or
            (args.radius is not None and (not np.isfinite(args.radius) or args.radius <= 0)) or
            (args.center is not None and not np.isfinite(args.center).all())):
        raise ValueError("--center requires a positive --radius, and vice versa")
    config = read_json(args.config)
    if config["sample_m"] <= 0 or config["raster_resolution_m"] <= 0 or not 1 <= config["rounds"] <= 10:
        raise ValueError("Invalid spacing, resolution or rounds")
    if config["halo_m"] < config["max_gap_m"] + 30:
        raise ValueError("halo_m must cover max_gap_m plus 30m context")
    inputs = discover(args.project, args.period, args.area)
    before = fingerprint(inputs)
    frame = load_lines(inputs)
    if frame.empty:
        raise ValueError("No formal lines")
    if args.inspect:
        print(f"Area={inputs['area']} period={args.period} metric_crs={frame.crs}")
        print(f"Bounds={frame.total_bounds.tolist()} formal_features={len(frame)}")
        print(f"Missing caches={inputs['missing']}")
        return
    started = time.perf_counter()
    if args.bbox:
        if args.bbox[0] >= args.bbox[2] or args.bbox[1] >= args.bbox[3] or not np.isfinite(args.bbox).all():
            raise ValueError("Invalid metric --bbox")
        roi = box(*args.bbox)
    elif args.center:
        x, y = args.center
        roi = box(x-args.radius, y-args.radius, x+args.radius, y+args.radius)
    else:
        print("Selecting an 800m candidate-dense ROI (land use unverified)", flush=True)
        roi = select_roi(frame, config)
    context = roi.buffer(config["halo_m"], cap_style="square")
    # Keep whole features for topology. Evidence is loaded only for ROI + halo.
    local = frame[frame.intersects(context)].copy()
    if local.empty:
        raise ValueError("ROI does not intersect formal centerlines; coordinates must use printed metric CRS")
    print(f"ROI {list(roi.bounds)}; {len(local)} context features", flush=True)
    edges = build(local)
    evidence = Evidence(inputs, frame.crs, context, config)
    root = ROOT / "outputs"
    if not root.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("outputs must resolve inside the experiment directory")
    root.mkdir(exist_ok=True)
    output = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    output.mkdir()
    provenance = dict(project=str(inputs["project"]), period=args.period, area=inputs["area"],
                      metric_crs=str(frame.crs), bbox_m=list(roi.bounds), config=config,
                      source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in ROOT.glob("*.py")},
                      roi_selection="explicit" if args.bbox or args.center else "candidate_density_landuse_unverified",
                      missing_caches=inputs["missing"], input_fingerprints=before,
                      original_images=inputs["images"], probability_sources=inputs["probabilities"],
                      local_source_index_map={i: int(row.source_id) for i, (_, row) in enumerate(local.iterrows())},
                      molra_sources=inputs["molra"],
                      formal_surface_is_independent_evidence=False,
                      algorithms_reused=["canonical_road_surface._build_graph/_collapse_duplicates/_outward",
                        "road_connection_evidence.ConnectionEvidence.measure/RoadProbability/ArrayRoadProbability",
                        "road_axis_quality._evidence_route", "fast_image_structure.normalized_gray/gradients"])
    write_json(output / "inputs.json", provenance)
    summary, audits = {}, {}
    print("Experiment 0: baseline", flush=True)
    summary["baseline"] = save_result(output / "baseline", edges, [], [], local, evidence, roi, frame.crs, config, "A: formal baseline")
    # Baseline export is the actual formal features, with clipping only.
    (output / "baseline/centerlines.gpkg").unlink()
    save_lines(output / "baseline/centerlines.gpkg", local, roi)
    experiments = [("geometry_only", "geometry_only", ("spur", "gap", "duplicate")),
                   ("single_spur", "image_guided", ("spur",)),
                   ("single_gap", "image_guided", ("gap",)),
                   ("single_duplicate", "image_guided", ("duplicate",)),
                   ("image_guided", "image_guided", ("spur", "gap", "duplicate"))]
    for name, mode, stages in experiments:
        print(f"Running {name}", flush=True)
        result, audit, changes = run(edges, evidence, roi, config, mode, stages)
        summary[name] = save_result(output / name, result, audit, changes, local, evidence, roi, frame.crs, config, name)
        if name.startswith("single_"):
            candidate_sheets(output / name / "candidate_reviews", audit, local, evidence)
        elif name == "image_guided":
            candidate_sheets(output / name / "accepted_edits", [r for r in audit if r["decision"] in ("delete", "connect", "merge")], local, evidence)
        audits[name] = audit
    print("Diagnosing missing interior axes between road-side lines", flush=True)
    boundary_audit = diagnose(edges, evidence, roi)
    audits["boundary_axis_diagnostic"] = boundary_audit
    diagnostic = output / "boundary_axis_diagnostic"
    diagnostic.mkdir()
    write_json(diagnostic / "audit.json", boundary_audit)
    ribbon_overview(diagnostic / "review.png", boundary_audit, local, evidence, roi)
    candidate_sheets(diagnostic / "candidate_reviews", boundary_audit, local, evidence)
    if boundary_audit:
        import geopandas as gpd
        from shapely import wkt
        proposals = gpd.GeoDataFrame(dict(candidate_id=[r["candidate_id"] for r in boundary_audit],
                                          status=["review_only"]*len(boundary_audit)),
            geometry=[wkt.loads(r["result_wkt"]) for r in boundary_audit], crs=frame.crs)
        proposals.to_file(diagnostic / "axis_hypotheses.gpkg", layer="review_only", driver="GPKG")
    summary["boundary_axis_diagnostic"] = dict(candidate_count=len(boundary_audit), auto_applied=0)
    after = fingerprint(inputs)
    if before != after:
        write_json(output / "input_change_warning.json", dict(before=before, after=after))
        raise RuntimeError("Inputs changed during experiment (possibly another process); comparison invalid")
    comparison = output / "comparison"
    comparison.mkdir()
    numeric = [k for k, v in summary["baseline"].items() if isinstance(v, (float, int)) or v is None]
    table = "| Metric | A baseline | B geometry | C image | C minus B |\n|---|---:|---:|---:|---:|\n"
    for key in numeric:
        a, b, c = [summary[n].get(key) for n in ("baseline", "geometry_only", "image_guided")]
        delta = c-b if c is not None and b is not None else None
        table += f"| {key} | {a} | {b} | {c} | {delta} |\n"
    (comparison / "comparison.md").write_text(table + "\nTopology counts and image scores are diagnostics, not ground-truth accuracy. Human review is required.\n", encoding="utf8")
    write_json(output / "audit.json", dict(provenance=provenance, experiments=audits,
        input_fingerprints_unchanged=True, deferred=["junction_rewire", "loop_reconstruction", "partial_duplicate_rewire"]))
    write_json(output / "summary.json", dict(experiments=summary, elapsed_seconds=time.perf_counter()-started,
        input_fingerprints_unchanged=True, accuracy_claim="Unlabelled ROI; no precision/recall claim"))
    print(f"Completed: {output}", flush=True)
    return output


if __name__ == "__main__":
    main()

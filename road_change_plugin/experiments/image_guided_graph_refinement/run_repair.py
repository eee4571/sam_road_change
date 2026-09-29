"""Actual experimental repair. Inputs are read-only; all products stay in outputs."""
import argparse
from datetime import datetime, timezone
import hashlib
from uuid import uuid4
import time

import numpy as np
from shapely.geometry import box

from bootstrap import ROOT
from data import discover, load_lines, fingerprint, read_json
from evidence import Evidence
from graph import build
from medial_repair import propose_axes, replace_corridors, line_parts, parallel_ribbon_veto
from local_graph_repair import clean_local, smooth_local
from rgb_route import connect_gaps, ribbon_evidence
from report import write_json
from repair_artifacts import compare, final_products, topology, write_report
from run_experiment import select_roi


def repair_tile(frame, inputs, roi, config, output):
    output.mkdir()
    evidence = Evidence(inputs, frame.crs, roi.buffer(160), config)
    before = topology(frame, roi, config)
    working, preliminary = clean_local(frame, evidence, roi)
    proposals, seeds = propose_axes(build(working[working.intersects(roi.buffer(160))]), evidence, roi)
    selected, protected = [], []
    for proposal in proposals:
        veto = parallel_ribbon_veto(proposal, working, evidence)
        if veto:
            protected.append(veto)
        else:
            selected.append(proposal)
    proposals = selected
    write_json(output/'seed_audit.json', seeds)
    write_json(output/'axis_searches.json', getattr(evidence, 'axis_searches', []))
    proposal_rows = []
    for p in proposals:
        old = frame[frame.intersects(p.envelope)].geometry.intersection(p.envelope)
        scored = [(g.length, ribbon_evidence(g, p.band_scale, evidence)['raw_rgb_support'])
                  for geom in old for g in line_parts(geom) if g.length >= 5]
        prior_score = sum(length*score for length, score in scored)/max(sum(length for length, _ in scored), 1)
        p.evidence.update(old_axis_raw_rgb_support=prior_score,
                          new_axis_raw_rgb_support=ribbon_evidence(p.axis, p.band_scale, evidence)['raw_rgb_support'])
        proposal_rows.append(dict(id=p.key, axis_wkt=p.axis.wkt, envelope_wkt=p.envelope.wkt,
            support=p.support, band_scale=p.band_scale, level=p.level, evidence=p.evidence, seed_ids=p.seed_ids))
    write_json(output/'proposals.json', proposal_rows)
    result, edits, contacts = replace_corridors(working, proposals, roi, evidence)
    audit = preliminary+edits+protected
    convergence = []
    for iteration in range(3):
        result, cleanup = clean_local(result, evidence, roi)
        result, gaps = connect_gaps(result, evidence, roi, config)
        changed = len(cleanup)+sum(r['action'] == 'connect' for r in gaps)
        convergence.append(dict(iteration=iteration+1, edits=changed))
        audit.extend(cleanup+gaps)
        if not changed:
            break
    result, smoothing = smooth_local(result, evidence, roi)
    audit.extend(smoothing)
    for contact in contacts:
        audit.append(dict(action='reattach_branch', new_wkt=contact['geometry_wkt'],
            evidence=contact, reason='preserve_external_branch_contact_at_rgb_medial_axis'))
    after = topology(result, roi, config)
    accepted = (after['invalid_features'] == 0 and after['zero_length_features'] == 0 and
                after['self_intersecting_features'] <= before['self_intersecting_features'] and
                after['component_count'] <= before['component_count'] and
                after['near_duplicate_length_m'] <= before['near_duplicate_length_m']+1.)
    if not accepted:
        for row in audit:
            row['applied'] = False
        audit.append(dict(action='rollback_tile', applied=False, reason='new_invalid_geometry_self_intersection_component_split_or_duplicate_increase'))
        result = frame.copy()
    else:
        for row in audit:
            row.setdefault('applied', row['action'] != 'retain_gap')
    write_json(output/'branch_reattachments.json', contacts)
    write_json(output/'audit.json', audit)
    summary = dict(bbox_m=list(roi.bounds), roi_wkt=roi.wkt, accepted=accepted,
        medial_axes=len(proposals), convergence=convergence, before=before,
        proposed_after=after, after=after if accepted else before)
    write_json(output/'summary.json', summary)
    compare(output/'before_after.png', frame, result, evidence, roi, 'Local repair: yellow centerlines over raw RGB')
    centers = []
    for p in proposals:
        for q in proposals:
            hit = p.axis.intersection(q.axis)
            if p is not q and hit.geom_type == 'Point' and all(hit.distance(c) > 100 for c in centers):
                centers.append(hit)
    if not centers and proposals:
        centers = [max(proposals, key=lambda p: p.axis.length).axis.interpolate(.5, normalized=True)]
    for i, center in enumerate(centers[:4]):
        close = box(center.x-120, center.y-120, center.x+120, center.y+120)
        compare(output/f'local_{i+1:02d}.png', frame, result, evidence, close, 'Junction / central axis detail')
    return result, audit, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--period', required=True)
    parser.add_argument('--area')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--bbox', nargs=4, type=float)
    group.add_argument('--center', nargs=2, type=float)
    parser.add_argument('--radius', type=float)
    parser.add_argument('--whole-period', action='store_true', help='Scan bounded tiles after the selected priority ROI')
    parser.add_argument('--tile-size', type=float, default=900.)
    args = parser.parse_args(argv)
    if ((args.center is None) != (args.radius is None) or
            (args.radius is not None and (not np.isfinite(args.radius) or args.radius <= 0)) or
            (args.center is not None and not np.isfinite(args.center).all())):
        parser.error('--center and positive --radius must be specified together')
    if not 400 <= args.tile_size <= 1500:
        parser.error('--tile-size must be between 400 and 1500 metres')
    inputs = discover(args.project, args.period, args.area)
    fingerprints = fingerprint(inputs)
    frame = load_lines(inputs)
    config = read_json(ROOT/'config.json')
    if args.bbox:
        if not np.isfinite(args.bbox).all() or args.bbox[0] >= args.bbox[2] or args.bbox[1] >= args.bbox[3]:
            parser.error('Invalid metric bbox')
        roi = box(*args.bbox)
    elif args.center:
        x, y = args.center
        roi = box(x-args.radius, y-args.radius, x+args.radius, y+args.radius)
    else:
        roi = select_roi(frame, config)
    if not frame.intersects(roi).any():
        parser.error('ROI does not intersect input centerlines')
    root = ROOT/'outputs'
    if not root.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Outputs must resolve inside the experiment directory')
    output = root/('repair_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    output.mkdir(parents=True)
    started = time.perf_counter()
    write_json(output/'inputs.json', dict(project=inputs['project'], period=args.period, area=inputs['area'],
        priority_bbox_m=list(roi.bounds), metric_crs=str(frame.crs), config=config, input_fingerprints=fingerprints,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},
        missing=inputs['missing'], full_period_output=True, whole_period_scan=args.whole_period,
        source_images=inputs['images'], source_probabilities=inputs['probabilities'], source_molra=inputs['molra']))
    regions = [roi]
    if args.whole_period:
        x0, y0, x1, y1 = frame.total_bounds
        for y in np.arange(y0-1, y1, args.tile_size):
            for x in np.arange(x0-1, x1, args.tile_size):
                tile = box(x, y, x+args.tile_size, y+args.tile_size).difference(roi)
                if not tile.is_empty and frame.intersects(tile).any():
                    regions.append(tile)
    result, audit, tiles = frame.copy(), [], []
    for i, region in enumerate(regions):
        print(f'Tile {i+1}/{len(regions)}: {list(region.bounds)}', flush=True)
        result, edits, tile = repair_tile(result, inputs, region, config, output/f'tile_{i:03d}')
        for row in edits:
            row['tile'] = i
        audit.extend(edits)
        tiles.append(tile)
        print(f'  accepted={tile["accepted"]}; axes={tile["medial_axes"]}; loops={tile["before"]["small_loop_count"]}->{tile["after"]["small_loop_count"]}', flush=True)
    summary = final_products(output, frame, result, inputs, config, audit, tiles)
    if fingerprints != fingerprint(inputs):
        raise RuntimeError('Input fingerprints changed during the experiment')
    summary.update(input_fingerprints_unchanged=True, elapsed_seconds=time.perf_counter()-started)
    write_json(output/'summary.json', summary)
    write_report(output, summary)
    print(output, flush=True)
    return output


if __name__ == '__main__':
    main()

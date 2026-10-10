"""Revalidate saved graph proposals after a checker fix, without repeating search."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
from uuid import uuid4
import geopandas as gpd
from shapely.geometry import box

from bootstrap import ROOT
from data import read_json, discover, fingerprint
from run_multi_roi import data_fingerprints
from evidence import Evidence
from graph_transaction import validate
from repair_artifacts import save_roi_result
from multi_roi_report import operation_details, write_multi_report
from report import write_json


def revalidate(source):
    source = Path(source).resolve()
    if not source.is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('Only existing experiment outputs can be revalidated')
    inputs_meta = read_json(source/'inputs.json')
    selection = read_json(source/'selection.json')
    inputs = discover(inputs_meta['project'], inputs_meta['period'], inputs_meta['area'],
                      record=inputs_meta['selected_period_record'])
    start = fingerprint(inputs)
    if data_fingerprints(start, inputs['assets']['manifest']) != data_fingerprints(inputs_meta['fingerprints'], inputs['assets']['manifest']):
        raise ValueError('Cached proposals are bound to different image/vector inputs')
    output = ROOT/'outputs'/('block_verified_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    output.mkdir(); (output/'source_snapshot').mkdir()
    print(output, flush=True)
    metadata = dict(inputs_meta, fingerprints=start, cached_proposals_from=str(source),
                    proposal_generation_source_sha256=inputs_meta['source_sha256'],
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')})
    write_json(output/'inputs.json', metadata)
    for p in ROOT.glob('*.py'):
        shutil.copy2(p, output/'source_snapshot'/p.name)
    for name in ('selection.json', 'roi_overview.png'):
        shutil.copy2(source/name, output/name)
    summaries = []
    for row in selection['regions']:
        old = source/row['id']; folder = output/row['id']; folder.mkdir()
        roi = box(*row['bbox_m'])
        before = gpd.read_file(old/'baseline_centerlines.gpkg', layer='context')
        current = gpd.read_file(old/'final_centerlines.gpkg', layer='context')
        proposed = gpd.read_file(old/'block_proposal.gpkg', layer='centerlines')
        audit = read_json(old/'audit.json')
        evidence = Evidence(inputs, before.crs, roi.buffer(160), inputs_meta['config'])
        evidence.multichannel_ribbon = True; evidence.independent_flanks = True
        last = next(a for a in reversed(audit) if a['action'] == 'block_joint_fit')
        # Already accepted products are retained; rejected final proposals are
        # reconsidered against precisely their saved pre-fit graph.
        if not last['applied']:
            guard = validate(current, proposed, roi, evidence)
            last['previous_guard'] = last['transaction_guard']
            last.update(transaction_guard=guard, applied=guard['passed'],
                        revalidation_reason='both_graphs_use_identical_millimetre_precision_for_loop_detection')
            if guard['passed']:
                current = proposed
        print(row['id'], 'joint fit:', last['applied'], last['transaction_guard']['failures'], flush=True)
        for name in ('initial.png', 'axis_searches.json', 'block_proposal.gpkg'):
            shutil.copy2(old/name, folder/name)
        summary = save_roi_result(folder, before, current, evidence, roi, inputs_meta['config'], row, audit,
                                  [dict(stage='cached_candidate_revalidation', accepted=last['applied'])])
        operation_details(folder, before, current, evidence, roi, audit)
        summaries.append(summary); write_json(output/'summary.json', summaries)
    final = fingerprint(inputs)
    unchanged = data_fingerprints(start, inputs['assets']['manifest']) == data_fingerprints(final, inputs['assets']['manifest'])
    write_json(output/'verification.json', dict(input_fingerprints_unchanged=start == final,
        input_assets_unchanged=unchanged, independent_roi_baselines=True, no_models_or_width_recomputed=True,
        candidate_search_reused=True))
    if not unchanged:
        raise ValueError('Actual inputs changed during revalidation')
    write_multi_report(output, summaries)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    revalidate(parser.parse_args().run_directory)

"""Independent multi-ROI experiment with common parameters and untouched controls."""
import argparse
from datetime import datetime, timezone
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

from bootstrap import ROOT, PLUGIN
from data import discover, load_lines, fingerprint, read_json
from evidence import Evidence, ImageGrid
from report import background, draw, write_json
from repair_artifacts import save_roi_result
from roi_mining import select_regions
from multi_roi_report import operation_details, selection_map, write_multi_report


def data_fingerprints(fingerprints, manifest_path):
    """Discovery metadata can change for another period; actual inputs cannot."""
    return {path: value for path, value in fingerprints.items() if Path(path) != Path(manifest_path)}


def imported_project_sources():
    """Record only already-imported project dependencies; never scan/load models."""
    root = (PLUGIN/'code').resolve()
    sources = {}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if not name:
            continue
        path = Path(name).resolve()
        if path.suffix == '.py' and path.is_relative_to(root) and path.is_file():
            sources[str(path.relative_to(root))] = path.read_bytes()
    return sources


def initial_figure(path, frame, evidence, roi, title):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax in axes:
        background(ax, evidence)
        x0, y0, x1, y1 = roi.bounds
        ax.set(xlim=(x0, x1), ylim=(y0, y1), aspect='equal')
        ax.ticklabel_format(style='plain', useOffset=False)
    for line in frame[frame.intersects(roi)].geometry:
        draw(axes[1], line, color='#ffff33', linewidth=.9)
    axes[0].set_title('Raw RGB')
    axes[1].set_title('Current formal network')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--period', required=True)
    parser.add_argument('--area')
    parser.add_argument('--algorithm', choices=('local', 'block', 'structural', 'medial'), default='local')
    parser.add_argument('--roi-size', type=float, default=600.)
    parser.add_argument('--count', type=int, default=9, choices=range(6, 11))
    parser.add_argument('--selection-only', action='store_true')
    parser.add_argument('--selection', type=Path, help='Reuse a saved selection.json; does not reuse modified roads')
    parser.add_argument('--inputs', type=Path, help='Bind to a previous experiment inputs.json; verify all asset fingerprints')
    parser.add_argument('--only', nargs='+', help='Subset of saved ROI IDs for iteration')
    args = parser.parse_args(argv)
    if not 400 <= args.roi_size <= 800:
        parser.error('--roi-size must be 400..800m')
    saved_inputs = read_json(args.inputs) if args.inputs else None
    if saved_inputs and Path(saved_inputs['project']).resolve() != Path(args.project).resolve():
        raise ValueError('Saved inputs belong to a different project')
    inputs = discover(args.project, args.period, args.area,
                      record=saved_inputs['selected_period_record'] if saved_inputs else None)
    before_fingerprint = fingerprint(inputs)
    if saved_inputs and data_fingerprints(saved_inputs['fingerprints'], inputs['assets']['manifest']) != data_fingerprints(before_fingerprint, inputs['assets']['manifest']):
        raise ValueError('Saved experiment actual assets changed; refusing mixed-input comparison')
    frame = load_lines(inputs)
    dependencies = imported_project_sources()
    config = read_json(ROOT/'config.json')
    output = ROOT/'outputs'/('multi_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    if not output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Output must be inside experiment directory')
    output.mkdir(parents=True)
    manifest = dict(project=inputs['project'], period=args.period, area=inputs['area'], crs=str(frame.crs),
        baseline='current formal centerlines; every ROI starts independently from the same baseline', algorithm=args.algorithm,
        fingerprints=before_fingerprint, selected_period_record=inputs['record'], bound_input_manifest=str(args.inputs) if args.inputs else None, config=config, missing=inputs['missing'],
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')})
    manifest['dependency_sha256'] = {name: hashlib.sha256(value).hexdigest() for name, value in dependencies.items()}
    write_json(output/'inputs.json', manifest)
    snapshot = output/'source_snapshot'
    snapshot.mkdir()
    for source in ROOT.glob('*.py'):
        (snapshot/source.name).write_bytes(source.read_bytes())
    for name, value in dependencies.items():
        destination = snapshot/'project_dependencies'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(value)
    print(output, flush=True)
    if args.selection:
        saved = read_json(args.selection)
        if (data_fingerprints(saved['fingerprints'], inputs['assets']['manifest']) !=
                data_fingerprints(before_fingerprint, inputs['assets']['manifest']) or saved['crs'] != str(frame.crs)):
            raise ValueError('Saved selection is bound to different inputs or CRS')
        regions = saved['regions']
    else:
        print('Geometry ROI mining with coarse RGB coverage/scene filter', flush=True)
        overview = ImageGrid(inputs['images'], frame.crs, box(*frame.total_bounds).buffer(5), 5.)
        regions, mining = select_regions(frame, config, args.roi_size, args.count, overview)
        write_json(output/'mining.json', mining)
    if args.only:
        regions = [r for r in regions if r['id'] in args.only]
        if len(regions) != len(set(args.only)):
            parser.error('Unknown ROI ID')
    write_json(output/'selection.json', dict(regions=regions, fingerprints=before_fingerprint, crs=str(frame.crs)))
    selection_map(output, frame, inputs, regions)
    summaries = []
    for row in regions:
        folder = output/row['id']
        folder.mkdir()
        roi = box(*row['bbox_m'])
        print(f"{row['id']} {row['category']} {row['bbox_m']}", flush=True)
        local = frame[frame.intersects(roi.buffer(160))].copy().reset_index(drop=True)
        if args.selection_only:
            evidence = SimpleNamespace(image=ImageGrid(inputs['images'], frame.crs, roi, 1.))
            initial_figure(folder/'initial.png', local, evidence, roi, row['id']+' / '+row['category'])
            continue
        if args.algorithm == 'structural':
            from structural_refinement import optimize
        elif args.algorithm == 'block':
            from block_graph import optimize
        else:
            from local_graph_optimizer import optimize
        evidence = Evidence(inputs, frame.crs, roi.buffer(160), config)
        evidence.debug_output = folder
        initial_figure(folder/'initial.png', local, evidence, roi, row['id']+' / '+row['category'])
        if args.algorithm == 'medial':
            from local_graph_optimizer import medial_steps
            evidence.multichannel_ribbon = True
            evidence.independent_flanks = True
            audit, convergence = [], []
            result = medial_steps(local, evidence, roi, audit)
        else:
            result, audit, convergence = optimize(local, evidence, roi, config)
        write_json(folder/'axis_searches.json', getattr(evidence, 'axis_searches', []))
        summary = save_roi_result(folder, local, result, evidence, roi, config, row, audit, convergence)
        operation_details(folder, local, result, evidence, roi, audit)
        summaries.append(summary)
        write_json(output/'summary.json', summaries)
        print(f"  applied={sum(a.get('applied',False) for a in audit)}; loops={summary['before']['small_loop_count']}->{summary['after']['small_loop_count']}", flush=True)
    after_fingerprint = fingerprint(inputs)
    assets_unchanged = (data_fingerprints(before_fingerprint, inputs['assets']['manifest']) ==
                        data_fingerprints(after_fingerprint, inputs['assets']['manifest']))
    write_json(output/'verification.json', dict(input_fingerprints_unchanged=before_fingerprint == after_fingerprint,
        input_assets_unchanged=assets_unchanged, no_models_or_width_recomputed=True, independent_roi_baselines=True,
        imported_project_sources_unchanged=all((PLUGIN/'code'/name).read_bytes() == value for name, value in dependencies.items()),
        fingerprint_changes={p: dict(before=before_fingerprint.get(p), after=after_fingerprint.get(p))
                             for p in set(before_fingerprint)|set(after_fingerprint)
                             if before_fingerprint.get(p) != after_fingerprint.get(p)}))
    if not assets_unchanged:
        raise RuntimeError('Input files changed during experiment')
    if summaries:
        write_multi_report(output, summaries)
    return output


if __name__ == '__main__':
    main()

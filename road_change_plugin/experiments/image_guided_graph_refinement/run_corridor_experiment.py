"""Whole-road RGB corridor experiment; reads cached products, writes only outputs."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
from uuid import uuid4
from types import SimpleNamespace
import copy
import shutil
import hashlib
import geopandas as gpd
from shapely import union_all,points,distance,from_wkt
import numpy as np
from shapely.geometry import box
from bootstrap import ROOT
from data import discover,load_lines,read_json,fingerprint
from evidence import Evidence,stations
from report import write_json
from audit_extraction_stages import figure
from whole_corridor import proposals,axis_strength
from medial_repair import replace_corridors,parallel_ribbon_veto,line_parts
from graph_transaction import validate
from repair_artifacts import save_roi_result


def output_directory(prefix):
    destination=ROOT/'outputs'/(prefix+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    if not destination.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Experiment output must stay inside the experiment directory')
    destination.mkdir(parents=True)
    return destination


def replaced_extent_distance(current,candidate,owned):
    """Measure lost road extent even when the surviving graph stays connected."""
    probes=[stations(g,3)[1] for line in current.geometry.intersection(owned)
            for g in line_parts(line) if g.length>1]
    if not probes:return 0.
    if candidate.empty:return float('inf')
    return float(distance(points(np.concatenate(probes)),union_all(candidate.geometry)).max())


def road_units(proposals):
    roads={}
    for proposal in proposals:
        roads.setdefault(proposal.evidence.get('corridor_id',proposal.key),[]).append(proposal)
    return list(roads.values())


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',required=True);p.add_argument('--period',required=True)
    p.add_argument('--area',default='1');p.add_argument('--inputs',type=Path)
    region=p.add_mutually_exclusive_group(required=True)
    region.add_argument('--bbox',nargs=4,type=float)
    region.add_argument('--roi-list',type=Path,help='JSON regions with id/category/bbox_m; geometry selection only')
    args=p.parse_args(argv)
    if args.roi_list:
        regions=read_json(args.roi_list)['regions']
        if not regions or any(len(r['bbox_m'])!=4 for r in regions):raise ValueError('Invalid ROI list')
        batch=output_directory('corridor_multi_');summaries=[]
        print('BATCH',batch,flush=True)
        for row in regions:
            identifier=row['id']
            if not identifier or not all(c.isalnum() or c=='_' for c in identifier):raise ValueError('Invalid ROI id')
            arguments=['--project',args.project,'--period',args.period,'--area',args.area,
                       '--bbox',*[str(x) for x in row['bbox_m']]]
            if args.inputs:arguments+=['--inputs',str(args.inputs)]
            print('ROI',identifier,flush=True)
            child=main(arguments)
            # One shared source snapshot avoids redundant deep Windows paths.
            shutil.copytree(child,batch/identifier,ignore=shutil.ignore_patterns('source_snapshot'))
            summary=read_json(child/'summary.json');summary.update(row)
            write_json(batch/identifier/'summary.json',summary)
            summaries.append(summary)
            write_json(batch/'summary.json',summaries)
            manifest=read_json(child/'inputs.json')
            if len(summaries)==1:
                write_json(batch/'inputs.json',manifest)
                shutil.copytree(child/'source_snapshot',batch/'source_snapshot')
            elif read_json(batch/'inputs.json')['fingerprints']!=manifest['fingerprints']:
                raise RuntimeError('Inputs changed between ROIs')
        write_json(batch/'selection.json',dict(regions=regions))
        from assemble_multi_roi import assemble
        assemble(batch)
        return batch
    saved=read_json(args.inputs) if args.inputs else None
    if saved and (Path(saved['project']).resolve()!=Path(args.project).resolve()
                  or str(saved['period'])!=args.period or str(saved['area'])!=args.area):
        raise ValueError('Saved discovery record belongs to a different project/period/area')
    record=saved['selected_period_record'] if saved else None
    inputs=discover(args.project,args.period,args.area,record=record)
    source_bytes={p.name:p.read_bytes() for p in ROOT.glob('*.py')}
    before=fingerprint(inputs);frame=load_lines(inputs);roi=box(*args.bbox)
    frame=frame[frame.intersects(roi.buffer(100))].copy().reset_index(drop=True)
    config=read_json(ROOT/'config.json')
    evidence=Evidence(inputs,frame.crs,roi.buffer(100),config)
    evidence.multichannel_ribbon=True;evidence.independent_flanks=True
    output=output_directory('corridor_');print(output,flush=True)
    write_json(output/'inputs.json',dict(project=inputs['project'],period=args.period,area=args.area,
        crs=str(frame.crs),bbox=args.bbox,fingerprints=before,selected_period_record=inputs['record']))
    fitted,searches=proposals(frame,evidence,roi)
    print('corridor proposals',len(fitted),flush=True)
    write_json(output/'searches.json',searches)
    proposal_frame=gpd.GeoDataFrame([dict(geometry=p.axis,scale=p.band_scale,score=p.support) for p in fitted],geometry='geometry',crs=frame.crs) if fitted else frame.iloc[:0]
    proposal_frame.to_file(output/'proposals.gpkg',layer='axes',driver='GPKG')
    figure(output/'proposals.png',[None,frame,proposal_frame],evidence,roi,['Raw RGB','Formal input','Joint corridor proposals'])
    result=frame.copy();audit=[];viable=[]
    for road in road_units(fitted):
        proposal=road[0]
        scale=proposal.evidence.get('road_aperture_m',proposal.band_scale)
        axis=from_wkt(proposal.evidence['road_axis_wkt']) if len(road)==2 else proposal.axis
        envelope=union_all([p.envelope for p in road])
        old_parts=[p for g in frame.geometry.intersection(envelope) for p in line_parts(g) if p.length>=15]
        old_strength=sum(p.length*axis_strength(p,scale,evidence)[0] for p in old_parts)/max(1,sum(p.length for p in old_parts))
        new_strength,new_valid=axis_strength(axis,scale,evidence)
        for track in road:
            track.evidence.update(old_axis_strength=old_strength,new_axis_strength=new_strength,
                comparable_rgb_gain=new_strength-old_strength,rgb_comparison_domain='whole_pavement_ribbon')
        # Repairing lost carriageway continuity need not improve a texture score:
        # a fragmented line may already sit on pavement. Accept a Pareto gain in
        # road representation only when RGB does not decline and the full ribbon
        # independently passed the unchanged support tests in fit_corridor.
        restores_pair=(len(road)==2 and proposal.evidence['paired_section_fraction']<.9
                       and new_strength>=old_strength)
        for track in road:track.evidence['selection_basis']='restore_pair_without_rgb_decline' if restores_pair else 'rgb_gain'
        if not old_parts or new_valid<.98 or (new_strength-old_strength<.05 and not restores_pair):
            audit.append(dict(action='keep_corridor',applied=False,candidate_id=proposal.key,
                evidence=proposal.evidence,reason='no_clear_improvement_under_identical_rgb_descriptor'))
            continue
        veto=parallel_ribbon_veto(proposal,result,evidence) if len(road)==1 else None
        if veto:
            audit.append(veto);continue
        viable.extend(road)
    def attempt(current,group):
        trials=copy.deepcopy(group)
        candidate,edits,contacts=replace_corridors(current,trials,roi,evidence)
        window=union_all([p.envelope.buffer(p.band_scale*2) for p in group]).intersection(roi)
        guard=validate(current,candidate,window,evidence)
        # A connected component may survive even when one of its real arms is
        # shortened drastically. Every replaced corridor station needs a nearby
        # replacement, separately from component-count and fixed-contact guards.
        owned=union_all([p.envelope for p in group])
        displacement=replaced_extent_distance(current,candidate,owned)
        guard['maximum_replaced_station_distance_m']=displacement
        if displacement>max(10.,max(p.band_scale for p in group)):
            guard['failures'].append('replaced_road_extent_lost')
            guard['passed']=False
        details=dict(action='whole_corridor_rebuild',candidate_ids=[p.key for p in group],
            applied=guard['passed'],evidence=[p.evidence for p in group],transaction_guard=guard,
            old_wkt=union_all([current.geometry.intersection(p.envelope).union_all() for p in group]).wkt,
            new_wkt=union_all([p.axis for p in trials]).wkt,
            suboperations=edits,branch_contacts=contacts,reason='joint_rgb_axis_replaces_same_road_fragments')
        audit.append(details)
        print('transaction',len(group),guard['failures'],flush=True)
        return (candidate.reset_index(drop=True),True) if guard['passed'] else (current,False)
    # Intersecting streets must replace the junction in one operation. Sequential
    # corridor cuts can manufacture a return loop before its other arm is rebuilt.
    groups=[]
    for proposal in viable:
        touching=[g for g in groups if any(proposal.envelope.intersects(q.envelope) for q in g)]
        groups=[g for g in groups if g not in touching]+[[proposal]+[q for g in touching for q in g]]
    for group in groups:
        result,applied=attempt(result,group)
        if not applied and len(group)>1:
            for road in road_units(group):result,_=attempt(result,road)
    save_roi_result(output,frame,result,evidence,roi,config,dict(id='corridor',category='whole road',bbox_m=args.bbox),audit,[])
    unchanged=before==fingerprint(inputs)
    write_json(output/'verification.json',dict(inputs_unchanged=unchanged,models_run=False,widths_recomputed=False))
    snapshot=output/'source_snapshot';snapshot.mkdir()
    for name,source in source_bytes.items():(snapshot/name).write_bytes(source)
    from run_multi_roi import imported_project_sources
    dependencies=imported_project_sources()
    dependency_index={}
    for name,source in dependencies.items():
        short=hashlib.sha256(name.encode()).hexdigest()[:16]+'.py'
        path=snapshot/'project_dependencies'/short;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(source)
        dependency_index[short]=dict(original_path=name,sha256=hashlib.sha256(source).hexdigest())
    write_json(snapshot/'project_dependencies/index.json',dependency_index)
    write_json(output/'source_hashes.json',{str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.rglob('*.py')})
    if not unchanged:raise RuntimeError('Input changed during experiment')
    return output


if __name__=='__main__':main()

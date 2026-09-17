"""Publication tiers from existing Fast2 interval evidence, without image rules."""
from collections import defaultdict
from collections import Counter
import time
import numpy as np
from shapely import from_wkt


class CandidateEvidence:
    def __init__(self,scenes,profiles,width_audit,absolute,relative):
        self.scenes=scenes;self.profiles=profiles or {};self.absolute=absolute;self.relative=relative
        self.width_rows={(r['axis_id'],round(r['start_m'],6),round(r['end_m'],6)):r for r in (width_audit or [])}
        self.targets={k:{r[2] for r in v} for k,v in self.profiles.items()}
        self.sources=defaultdict(set)
        self.cache={}
        for source,targets in self.targets.items():
            for target in targets:self.sources[target].add(source)

    def facts(self,row):
        if id(row) in self.cache:return self.cache[id(row)]
        kind=row['change_typ'];axis_id=row.get('source_axis');length=float(row['length_m'])
        facts={};self.cache[id(row)]=facts
        if kind in ('added','removed'):
            side=1 if kind=='added' else 0
            if axis_id is not None and self.scenes is not None:
                facts['uncovered_axis_fraction']=min(1.,length/max(self.scenes[side].lines[axis_id].length,1e-9))
            return facts
        key=(axis_id,round(row['start_m'],6),round(row['end_m'],6));audit=self.width_rows.get(key)
        if audit is None:return facts
        target=audit['target_axis']
        facts['one_to_one_match']=self.targets.get(axis_id)=={target} and self.sources[target]=={axis_id}
        facts['profile_interval_count']=audit['profile_interval_count']
        facts['quality_verified']=audit.get('quality_verified',False)
        facts['centerline_offset_m']=audit.get('centerline_offset_m',np.inf)
        for start,end,t,a,b,bw,aw in self.profiles.get(axis_id,[]):
            if t!=target or start>row['start_m']+1e-6 or end<row['end_m']-1e-6:continue
            span=np.maximum(0.,np.minimum(b,row['end_m'])-np.maximum(a,row['start_m']))
            threshold=np.maximum(self.absolute,self.relative*np.maximum(bw,aw))
            direction=1 if kind=='widened' else -1
            residual=direction*(aw-bw-float(row.get('v2_width_bias_m',0.)))
            strong=(residual>=2.5*threshold)&(residual>3*float(row.get('v2_profile_variability_m',np.inf)))
            # Contiguous coverage in already-built profile intervals, no sampling.
            longest=run=0.;previous=None
            for low,high,size,ok in zip(a,b,span,strong):
                if previous is not None and low>previous+1e-6:run=0.
                run=run+float(size) if ok and size>0 else 0.;longest=max(longest,run)
                previous=high
            facts['strong_profile_fraction']=longest/max(length,1e-9)
            facts['valid_profile_fraction']=float(span[np.isfinite(bw)&np.isfinite(aw)&(bw>0)&(aw>0)].sum())/max(length,1e-9)
            facts['matched_axis_fraction']=(end-start)/max(self.scenes[0].lines[axis_id].length,1e-9)
            break
        return facts

    def track_facts(self,row,facts):
        """Candidate-local vector checks; no raster reads or station rescan."""
        if self.scenes is None:return
        from .auto_track_evidence import displacement_rescue, nearby_opposite_support
        side=1 if row['change_typ']=='added' else 0
        axis=from_wkt(row['axis_wkt'])
        positions=np.linspace(0.,axis.length,5)
        source,target=self.scenes[side],self.scenes[1-side]
        displacement=displacement_rescue(axis,source,target,positions=positions)
        facts.update(displacement);reasons=[]
        if displacement['displacement_rescue']:reasons.append('same_road_displacement_rescue')
        if displacement['displacement_track_ambiguous']:reasons.append('cross_track_presence_ambiguity')
        if not reasons:
            nearby=nearby_opposite_support(axis,target,max(row['width_bef'],row['width_aft']),
                                          positions=positions,use_probability=False)
            facts.update(nearby)
            if nearby['nearby_opposite_reason']:reasons.append(nearby['nearby_opposite_reason'])
        facts['track_reasons']=reasons


def screen_candidates(records,evidence):
    """Qualify structure once before any expensive patch, keeping QA records."""
    started=time.perf_counter();counts=Counter();audit=[]
    for i,row in enumerate(records):
        if not row.get('v2_publish',False):
            counts['v2_structure_existing_'+row.get('v2_precision_reason','unknown')]+=1
            continue
        facts=evidence.facts(row)
        level,reasons,metrics=grade_candidate(row,'uncertain',facts,
            absolute=evidence.absolute,relative=evidence.relative)
        if not reasons and row['change_typ'] in ('added','removed'):
            evidence.track_facts(row,facts)
            level,reasons,metrics=grade_candidate(row,'uncertain',facts,
                absolute=evidence.absolute,relative=evidence.relative)
        row['v2_structure_passed']=not reasons
        if reasons:
            row.update(v2_publish=False,v2_publication_level='Candidate',
                v2_confidence_conditions=';'.join(reasons),v2_precision_reason='structure:'+(';'.join(reasons)))
            counts['v2_structure_rejected_'+row['change_typ']]+=1
            for reason in reasons:counts['v2_structure_reason_'+reason]+=1
            audit.append(dict(candidate=i,kind=row['change_typ'],state='structure_rejected',
                reasons=reasons,published=False,publication_level='Candidate',
                confidence_conditions=reasons,fast2_evidence=metrics,candidate_axis_wkt=row.get('axis_wkt'),
                candidate_geometry_wkb=row['geometry'].wkb_hex if 'geometry' in row else None))
        else:counts['v2_structure_passed_'+row['change_typ']]+=1
    counts['timing_v2_structure_seconds']=time.perf_counter()-started
    return dict(counts),audit


def grade_candidate(row,image_state,facts,*,absolute=2.,relative=.2):
    """Keep image uncertainty separate from the strength of the Auto candidate.

    Numeric margins only qualify publication; they never generate candidates,
    alter profiles, or interpret missing image evidence as a negative road.
    """
    if image_state=='strong_stable':return 'Rejected',[],{}
    if not row.get('v2_publish',False):return 'Candidate',['fast2_not_publishable'],{}
    kind=row['change_typ'];length=float(row.get('length_m',0.));fail=[];metrics=dict(facts)
    def check(name,ok):
        if not bool(ok):fail.append(name)
    if kind in ('added','removed'):
        source,target=('after','before') if kind=='added' else ('before','after')
        width=max(float(row.get('width_bef',0.)),float(row.get('width_aft',0.)))
        metrics.update(length_m=length,minimum_length=max(64.,8*width),
            source_surface=row.get(source+'_surface_ratio'),opposite_surface=row.get(target+'_surface_ratio'))
        check('presence_continuous_length',length>=max(64.,8*width))
        check('presence_valid_observation',row.get(source+'_valid_ratio',0.)>=.99 and row.get(target+'_valid_ratio',0.)>=.99)
        check('presence_source_support',row.get(source+'_geometry_ratio',0.)>=.99 and row.get(source+'_surface_ratio',0.)>=.75)
        check('presence_clear_opposite_absence',row.get('qa_state')=='confirmed' and
              row.get(target+'_surface_ratio',1.)<=.05 and row.get(target+'_probability_ratio',1.)==0.)
        check('presence_uncovered_segment',facts.get('uncovered_axis_fraction',0.)>=.5)
        # A long confirmed interval may cross a junction; junction is not a veto.
        check('presence_junction_context',not row.get('junction',False) or
              (length>=max(96.,12*width) and row.get('v2_temporal_state')=='persistent_change'))
        # Existence QA is not a correspondence ambiguity flag. Unknown ranking
        # must not be fabricated from a merely probable presence classification.
        check('presence_ambiguous_correspondence',not row.get('ambiguous',False))
        check('presence_temporal_stability',row.get('v2_temporal_state')!='transient_extraction_or_gap')
        for reason in facts.get('track_reasons',[]):check(reason,False)
    else:
        before=float(row.get('width_bef',np.nan));after=float(row.get('width_aft',np.nan))
        residual=abs(float(row.get('v2_width_residual_m',np.nan)))
        variability=float(row.get('v2_profile_variability_m',np.inf))
        scatter=float(row.get('v2_width_background_scatter_m',np.inf))
        margin=2.5*max(absolute,relative*max(before,after))
        metrics.update(length_m=length,width_residual_m=residual,required_width_margin_m=margin,
                       profile_variability_m=variability,background_scatter_m=scatter)
        check('width_continuous_length',length>=96.)
        check('width_difference_margin',np.isfinite(residual) and residual>=margin)
        check('width_variability_margin',residual>3*variability and residual>=3*scatter)
        check('width_one_to_one_match',facts.get('one_to_one_match',False))
        check('width_matched_coverage',facts.get('matched_axis_fraction',0.)>=.75)
        check('width_sustained_profile',facts.get('strong_profile_fraction',0.)>=.9)
        check('width_valid_profile_coverage',facts.get('valid_profile_fraction',0.)>=.9)
        check('width_ab_quality',facts.get('quality_verified',False))
        check('width_centerline_offset',facts.get('centerline_offset_m',np.inf)<=min(3.,min(before,after)/2))
    return ('Candidate' if fail else 'Confirmed' if image_state=='strong_change' else 'Probable'),fail,metrics

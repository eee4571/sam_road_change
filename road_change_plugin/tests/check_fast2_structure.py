"""Backend smoke: structural gates and bounded patch scheduling, no models."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
from collections import Counter
import copy
import io
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT.parent/'code/tests')]
import engine
engine.__path__ = [str(ROOT/'code/engine')]
import numpy as np
from shapely.geometry import LineString, box
from shapely.strtree import STRtree
from engine.fast_candidate_publication import CandidateEvidence, grade_candidate, screen_candidates
from engine.fast_patch_verification import PatchVerifier
from engine.fast_auto_v2 import _reliable_width_at, merge_presence_intervals, presence_publication, V2Config
from test_fast_patch_verification import DecisionTests as BaseDecisionTests, RasterTests
import user_pipeline  # pin the plugin entry before workbench fixtures change sys.path
import test_fast_auto_v2 as regression


def presence(kind='added'):
    row = dict(change_typ='added', length_m=150., width_bef=0., width_aft=8., v2_publish=True,
        qa_state='confirmed', junction=False, after_valid_ratio=1., before_valid_ratio=1.,
        after_geometry_ratio=1., after_surface_ratio=.9, before_surface_ratio=0., before_probability_ratio=0.,
        source_axis=0, start_m=0., end_m=150., axis_wkt='LINESTRING (0 0, 150 0)')
    if kind == 'removed':
        original=dict(row)
        for a,b in (('before','after'),('after','before')):
            for name in ('valid','geometry','surface','probability'):
                row[a+'_'+name+'_ratio']=original.get(b+'_'+name+'_ratio',0.)
        row.update(change_typ=kind,width_bef=8.,width_aft=0.)
    return row


def width():
    return dict(change_typ='widened',length_m=150.,width_bef=8.,width_aft=16.,v2_publish=True,
        v2_width_residual_m=8.,v2_profile_variability_m=.5,v2_width_background_scatter_m=.5,
        source_axis=0,start_m=0.,end_m=150.,axis_wkt='LINESTRING (0 0, 150 0)')


def scene(lines=(), surface=None):
    lines=list(lines)
    return SimpleNamespace(lines=lines,tree=STRtree(lines),surface=lambda axis:surface if surface is not None else box(0,100,150,110),
                           probability=Mock())


class StructuralTests(unittest.TestCase):
    def test_two_period_junction_and_strong_surface_loss_reach_image_review(self):
        row=presence();row.update(length_m=40.,end_m=40.,axis_wkt='LINESTRING (0 0, 40 0)',
            junction=True,v2_temporal_state='temporal_context_unknown',qa_state='probable',before_surface_ratio=.12)
        self.assertTrue(presence_publication(row,V2Config())[0])
        counts,_=screen_candidates([row],CandidateEvidence([scene(),scene([LineString([(0,0),(40,0)])])],{},[],2.,.2))
        self.assertTrue(row['v2_publish'])
        self.assertEqual(counts['v2_structure_passed_added'],1)
        self.assertNotIn('presence_junction_context',row['v2_confidence_conditions'])

    def test_merge_only_sub_tolerance_cuts_before_evidence(self):
        self.assertEqual(merge_presence_intervals([(0,18),(20,40),(48,70)],3),[(0,40),(48,70)])
        row=presence();row['length_m']=31.9
        self.assertFalse(presence_publication(row,V2Config())[0])
        row['length_m']=32.
        self.assertTrue(presence_publication(row,V2Config())[0])

    def test_existing_interval_and_formal_output_contract(self):
        for name in ('test_stable_segments_do_not_run_match',
                     'test_presence_symmetry_and_flat_regular_geometry',
                     'test_width_changes_use_existing_profiles',
                     'test_qualification_does_not_enter_station_review',
                     'test_finalization_never_reenters_station_qualification'):
            with self.subTest(name=name), redirect_stdout(io.StringIO()):
                case=regression.V2Tests(name)
                try:
                    case.setUp()
                    original=case.fixture.scene
                    def graded(*args,**kwargs):
                        result=original(*args,**kwargs);result.widths['quality_grade']='A'
                        return result
                    case.fixture.scene=graded
                    getattr(case,name)()
                finally:case.doCleanups()

    def test_strong_change_never_bypasses_presence_conditions(self):
        cases={'before_valid_ratio':0.,'after_surface_ratio':.1,'before_surface_ratio':.8,
               'length_m':20.,'ambiguous':True,
               'v2_temporal_state':'transient_extraction_or_gap'}
        for key,value in cases.items():
            row=presence();row[key]=value
            self.assertEqual(grade_candidate(row,'strong_change',{'uncovered_axis_fraction':1.})[0],'Candidate',key)
        for kind in ('added','removed'):
            self.assertEqual(grade_candidate(presence(kind),'strong_change',{'uncovered_axis_fraction':1.})[0],'Confirmed')

    def test_strong_change_never_bypasses_width_conditions(self):
        facts=dict(one_to_one_match=True,matched_axis_fraction=1.,strong_profile_fraction=1.,quality_verified=True,
                   valid_profile_fraction=1.,centerline_offset_m=0.)
        self.assertEqual(grade_candidate(width(),'strong_change',facts)[0],'Confirmed')
        for key in facts:
            bad=dict(facts);bad[key]=10. if key=='centerline_offset_m' else 0.
            self.assertEqual(grade_candidate(width(),'strong_change',bad)[0],'Candidate',key)
        for key in ('v2_profile_variability_m','v2_width_background_scatter_m'):
            row=width();row[key]=5.
            self.assertIn('width_variability_margin',grade_candidate(row,'strong_change',facts)[1])

    def test_profile_gaps_and_split_merge_are_not_continuous_evidence(self):
        line=LineString([(0,0),(150,0)])
        profile=(0.,150.,0,np.array([0.,90.]),np.array([60.,150.]),np.full(2,8.),np.full(2,16.))
        audit=[dict(axis_id=0,target_axis=0,start_m=0.,end_m=150.,profile_interval_count=2,centerline_offset_m=1.)]
        facts=CandidateEvidence([scene([line,line]),scene([line])],{0:[profile],1:[profile]},audit,2.,.2).facts(width())
        self.assertFalse(facts['one_to_one_match'])
        self.assertAlmostEqual(facts['valid_profile_fraction'],.8)
        self.assertAlmostEqual(facts['strong_profile_fraction'],.4)

    def test_missing_width_quality_is_not_ab(self):
        self.assertIn('width_ab_quality',grade_candidate(width(),'strong_change',{})[1])

    def test_displacement_and_nearby_surface_use_vectors_only(self):
        axis=LineString([(0,0),(150,0)]);other=LineString([(0,6),(150,6)])
        source=scene([axis]);target=scene([other])
        row=presence();evidence=CandidateEvidence([target,source],{},[],2.,.2)
        counts,audit=screen_candidates([row],evidence)
        self.assertFalse(row['v2_publish'])
        self.assertIn('same_road_displacement_rescue',audit[0]['reasons'])
        target=scene([],box(-1,5,151,7));row=presence()
        counts,audit=screen_candidates([row],CandidateEvidence([target,source],{},[],2.,.2))
        self.assertIn('nearby_opposite_surface_support',audit[0]['reasons'])
        target.probability.sample_axis.assert_not_called()

    def test_parallel_conflict_remains_auditable(self):
        axis=LineString([(0,0),(150,0)])
        source=scene([axis,LineString([(0,5),(150,5)])])
        target=scene([LineString([(0,6),(150,6)])])
        row=presence()
        counts,audit=screen_candidates([row],CandidateEvidence([target,source],{},[],2.,.2))
        self.assertTrue(row['v2_publish'])
        self.assertIn('cross_track_presence_ambiguity',row['v2_confidence_conditions'])

    def test_rejected_candidates_never_read_patches_or_controls(self):
        verifier=PatchVerifier.__new__(PatchVerifier)
        verifier.scenes=[scene(),scene([LineString([(0,0),(150,0)])])]
        verifier.tiles=[];verifier.counts=Counter();verifier.audit=[]
        verifier.patch=Mock(side_effect=AssertionError('unexpected image work'))
        verifier.calibrate=Mock(side_effect=AssertionError('unexpected control work'))
        records=[dict(presence(),after_valid_ratio=0.) for _ in range(100)]
        counts=verifier.verify(records,[{'unused':'control'}])
        self.assertEqual(counts['v2_structure_reason_presence_valid_observation'],100)
        self.assertEqual(len(verifier.audit),100)
        self.assertTrue(all(not r['v2_publish'] for r in records))

    def test_only_passed_structure_reaches_patch_once(self):
        verifier=PatchVerifier.__new__(PatchVerifier)
        verifier.scenes=[scene(),scene([LineString([(0,0),(150,0)])])]
        verifier.tiles=[];verifier.counts=Counter();verifier.audit=[]
        verifier.patch=Mock(return_value={});verifier.calibrate=Mock();verifier.record_decision=Mock()
        verifier.verify([dict(presence(),before_valid_ratio=0.),presence()],[])
        verifier.patch.assert_called_once()
        verifier.record_decision.assert_called_once()


class DecisionTests(BaseDecisionTests):
    def test_uncertain_without_strong_fast2_evidence_is_candidate(self):
        p=self.patch();p['valid']=False;v=self.verifier()
        for original in (True,False):
            row=dict(change_typ='added',v2_publish=original,v2_precision_reason='original_fast2')
            v.record_decision(row,p,0)
            self.assertEqual(row['v2_publish'],original)
            self.assertEqual(row['v2_publication_level'],'Candidate')
            self.assertEqual(row['qa_state'],'uncertain')

    def test_curved_or_unresolved_width_is_not_rejected_as_stable(self):
        p=self.stable();v=self.verifier()
        for axis,width in [('LINESTRING (0 0, 20 0, 20 20)',10),('LINESTRING (0 0, 40 0)',2)]:
            row=dict(change_typ='widened',v2_publish=True,v2_precision_reason='original',
                     width_bef=width,width_aft=width,axis_wkt=axis)
            v.record_decision(row,p,0)
            self.assertTrue(row['v2_publish'])
            self.assertEqual(row['v2_patch_state'],'uncertain')


del BaseDecisionTests
if __name__=='__main__':unittest.main()

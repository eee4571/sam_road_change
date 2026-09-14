import unittest
import numpy as np
from types import SimpleNamespace
from shapely.geometry import LineString
from engine.fast_candidate_publication import CandidateEvidence,grade_candidate


class PublicationTests(unittest.TestCase):
    def presence(self):
        return dict(change_typ='added',length_m=130,width_bef=0,width_aft=8,v2_publish=True,
            qa_state='confirmed',junction=False,after_valid_ratio=1,before_valid_ratio=1,
            after_geometry_ratio=1,after_surface_ratio=.9,before_surface_ratio=0,before_probability_ratio=0)

    def width(self):
        return dict(change_typ='widened',length_m=150,width_bef=8,width_aft=16,v2_publish=True,
            v2_width_residual_m=8,v2_profile_variability_m=.5,v2_width_background_scatter_m=.5)

    def test_four_levels(self):
        r=self.presence();facts={'uncovered_axis_fraction':1.}
        self.assertEqual(grade_candidate(r,'strong_change',{})[0],'Confirmed')
        self.assertEqual(grade_candidate(r,'strong_stable',facts)[0],'Rejected')
        self.assertEqual(grade_candidate(r,'uncertain',facts)[0],'Probable')
        r['after_surface_ratio']=.1
        self.assertEqual(grade_candidate(r,'uncertain',facts)[0],'Candidate')

    def test_no_promotion_of_upstream_rejection(self):
        r=self.presence();r['v2_publish']=False
        self.assertEqual(grade_candidate(r,'strong_change',{})[0],'Candidate')

    def test_presence_is_symmetric(self):
        a=self.presence();b=dict(a,change_typ='removed',width_bef=8,width_aft=0)
        for side,other in (('before','after'),('after','before')):
            for k in ('valid','surface','geometry','probability'):
                b[f'{side}_{k}_ratio']=a.get(f'{other}_{k}_ratio',0)
        self.assertEqual(grade_candidate(a,'uncertain',{'uncovered_axis_fraction':1})[:2],
                         grade_candidate(b,'uncertain',{'uncovered_axis_fraction':1})[:2])

    def test_each_presence_condition_is_auditable(self):
        cases=[('length_m',33,'presence_continuous_length'),
               ('after_surface_ratio',.2,'presence_source_support'),
               ('before_surface_ratio',.4,'presence_clear_opposite_absence'),
               ('before_valid_ratio',0,'presence_valid_observation')]
        for k,value,reason in cases:
            r=self.presence();r[k]=value
            self.assertIn(reason,grade_candidate(r,'uncertain',{'uncovered_axis_fraction':1})[1])
        r=self.presence();r.update(junction=True,v2_temporal_state='temporal_context_unknown')
        self.assertIn('presence_junction_context',grade_candidate(r,'uncertain',{'uncovered_axis_fraction':1})[1])
        r['v2_temporal_state']='persistent_change'
        self.assertEqual(grade_candidate(r,'uncertain',{'uncovered_axis_fraction':1})[0],'Probable')
        r['qa_state']='probable'
        self.assertNotIn('presence_ambiguous_correspondence',grade_candidate(r,'uncertain',{'uncovered_axis_fraction':1})[1])
        r['ambiguous']=True
        self.assertIn('presence_ambiguous_correspondence',grade_candidate(r,'uncertain',{'uncovered_axis_fraction':1})[1])

    def test_width_requires_continuous_reliable_profile(self):
        facts=dict(one_to_one_match=True,strong_profile_fraction=1.,matched_axis_fraction=.9)
        r=self.width();self.assertEqual(grade_candidate(r,'uncertain',facts)[0],'Probable')
        for key,reason in [('one_to_one_match','width_one_to_one_match'),
                           ('strong_profile_fraction','width_sustained_profile'),
                           ('matched_axis_fraction','width_matched_coverage')]:
            weak=dict(facts);weak[key]=0
            self.assertIn(reason,grade_candidate(r,'uncertain',weak)[1])
        r['v2_profile_variability_m']=4
        self.assertIn('width_variability_margin',grade_candidate(r,'uncertain',facts)[1])

    def test_context_uses_existing_profiles_and_detects_many_to_one(self):
        line=LineString([(0,0),(150,0)]);scenes=[SimpleNamespace(lines=[line,line]),None]
        profile=(0.,150.,4,np.array([0.,50.,100.]),np.array([50.,100.,150.]),np.full(3,8.),np.full(3,16.))
        audit=[dict(axis_id=0,start_m=0.,end_m=150.,target_axis=4,profile_interval_count=3)]
        r=self.width();r.update(source_axis=0,start_m=0.,end_m=150.)
        evidence=CandidateEvidence(scenes,{0:[profile]},audit,2.,.2)
        facts=evidence.facts(r)
        self.assertTrue(facts['one_to_one_match']);self.assertEqual(facts['strong_profile_fraction'],1.)
        evidence=CandidateEvidence(scenes,{0:[profile],1:[profile]},audit,2.,.2)
        self.assertFalse(evidence.facts(r)['one_to_one_match'])


if __name__=='__main__':unittest.main()

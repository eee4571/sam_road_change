import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from shapely.geometry import LineString
from shapely.strtree import STRtree
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine.auto_track_evidence import displacement_rescue, nearby_opposite_support

def scene(lines):return SimpleNamespace(lines=lines,tree=STRtree(lines))

class TrackEvidenceTests(unittest.TestCase):
    def test_small_displacement_with_high_overlap_is_rescued_both_directions(self):
        a=LineString([(0,0),(100,0)]); b=LineString([(0,5),(100,5)])
        for source,target in [(a,b),(b,a)]:
            result=displacement_rescue(source,scene([source]),scene([target]))
            self.assertTrue(result['displacement_rescue'])
            self.assertAlmostEqual(result['displacement_offset_m'],5.)

    def test_nearer_parallel_source_prevents_same_track_claim(self):
        a=LineString([(0,0),(100,0)]); b=LineString([(0,8),(100,8)])
        result=displacement_rescue(a,scene([a,b]),scene([b]))
        self.assertFalse(result['displacement_rescue'])
        self.assertTrue(result['displacement_track_ambiguous'])

    def test_diverging_and_transverse_tracks_are_not_rescued(self):
        a=LineString([(0,0),(100,0)])
        for b in [LineString([(0,1),(100,11)]),LineString([(50,-50),(50,50)]),LineString([(70,5),(100,5)])]:
            self.assertFalse(displacement_rescue(a,scene([a]),scene([b]))['displacement_rescue'])

    def test_nearby_parallel_surface_is_review_evidence_without_any_axis(self):
        axis=LineString([(0,0),(100,0)])
        surface=LineString([(0,8),(100,8)]).buffer(2)
        target=SimpleNamespace(surface=lambda a:surface,probability=None)
        self.assertEqual(nearby_opposite_support(axis,target,8)['nearby_opposite_reason'],'nearby_opposite_surface_support')
        target.surface=lambda a:LineString([(50,-50),(50,50)]).buffer(2)
        self.assertEqual(nearby_opposite_support(axis,target,8)['nearby_opposite_reason'],'')

if __name__=='__main__':unittest.main()

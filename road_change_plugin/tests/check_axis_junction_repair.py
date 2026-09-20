"""Junction-side repair must not cross a neighbouring road approach."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np
from shapely.geometry import LineString
from shapely.affinity import rotate,translate
from engine.road_axis_quality import repair_network_axes,axis_quality,_contacts


def junction():
    axis=LineString([(0,0),(-.3379,-1.5231),(4.8058,-1.9026),(5.3659,-2.1144),
        (5.3811,-2.0455),(6.4725,-2.3167),(6.7702,-2.3907),(7.2355,-2.6516),
        (7.2696,-2.498),(20.8091,-5.4073)])
    neighbour=LineString([(0,0),(5.3661,-2.1017),(5.367,-2.042),(.0258,.0499),
        (3.6376,7.0322),(6.1347,16.1093),(19.2599,43.2482)])
    return [axis,neighbour,LineString([(-15,0),(0,0)])]


class JunctionRepairTests(unittest.TestCase):
    def test_repair_preserves_contacts_and_station_mapping(self):
        for angle in (0,47,137):
            for reverse in (False,True):
                axes=[translate(rotate(a,angle,origin=(0,0)),400000,3500000) for a in junction()]
                if reverse:axes=[LineString(list(a.coords)[::-1]) for a in axes]
                with self.subTest(angle=angle,reverse=reverse):
                    repaired,audit=repair_network_axes(axes,[6.]*3,None)
                    self.assertFalse(axis_quality(repaired[0],6.)['abnormal'])
                    self.assertTrue(audit)
                    for i,a in enumerate(axes):
                        self.assertEqual(a.coords[0],repaired[i].coords[0])
                        self.assertEqual(a.coords[-1],repaired[i].coords[-1])
                        for j in range(i):
                            old=_contacts(a,axes[j]);new=_contacts(repaired[i],repaired[j])
                            self.assertTrue(new.difference(old.buffer(1e-5)).is_empty)
                            self.assertTrue(old.difference(new.buffer(1e-5)).is_empty)
                    for r in audit:
                        mapping=r['station_map']
                        self.assertTrue(np.all(np.diff(mapping['before'])>0))
                        self.assertTrue(np.all(np.diff(mapping['after'])>=0))
                        self.assertAlmostEqual(mapping['after'][-1],repaired[r['feature']].length)
                    self.assertTrue(repaired[2].equals_exact(axes[2],0))

    def test_no_repair_of_normal_junction(self):
        axes=[LineString([(0,0),(20,-5)]),LineString([(0,0),(5,15)]),LineString([(-15,0),(0,0)])]
        repaired,audit=repair_network_axes(axes,[6.]*3,None)
        self.assertFalse(audit)
        self.assertTrue(all(a.equals_exact(b,0) for a,b in zip(axes,repaired)))


if __name__=='__main__':unittest.main()

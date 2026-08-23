from __future__ import annotations

import sys
import unittest
from pathlib import Path

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString


WIDTH = Path(__file__).resolve().parents[1] / "engine" / "width"
sys.path.insert(0, str(WIDTH))

from road_change_detection import _analysis_crs  # noqa: E402


class LocalCrsCompatibilityTests(unittest.TestCase):
    def test_equivalent_named_local_crs_does_not_require_transformer(self) -> None:
        before_crs = CRS.from_wkt(
            'LOCAL_CS["CGCS2000_3_Degree_GK_CM_111E",UNIT["Meter",1],'
            'AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
        )
        after_crs = CRS.from_wkt(
            'LOCAL_CS["CGCS2000_GK_CM_111E",UNIT["Meter",1],'
            'AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
        )
        before = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (10, 0)])], crs=before_crs)
        after = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (11, 0)])], crs=after_crs)

        before_analysis, after_analysis, analysis_crs, output_crs = _analysis_crs(before, after)

        self.assertTrue(analysis_crs.is_engineering)
        self.assertTrue(before_analysis.crs.equals(output_crs))
        self.assertTrue(after_analysis.crs.equals(output_crs))
        self.assertEqual(tuple(before_analysis.geometry.iloc[0].coords)[-1], (10.0, 0.0))


if __name__ == "__main__":
    unittest.main()

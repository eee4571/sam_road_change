from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, box

import temporal_road_analysis


class TemporalRoadAnalysisTests(unittest.TestCase):
    def _period(self, root: Path, period: str, rows: list[tuple[LineString, float]]) -> Path:
        path = root / f"{period}.shp"
        gpd.GeoDataFrame(
            {
                "global_id": list(range(len(rows))),
                "width_map": [width for _geometry, width in rows],
            },
            geometry=[geometry for geometry, _width in rows],
            crs="EPSG:3857",
        ).to_file(path, encoding="UTF-8")
        return path

    def test_builds_shp_lifecycle_observations_and_events(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            road_a = LineString([(0, 0), (100, 0)])
            road_b = LineString([(0, 30), (100, 30)])
            paths = {
                "2021": self._period(root, "2021", [(road_a, 4.0)]),
                "2022": self._period(root, "2022", [(road_a, 4.0), (road_b, 5.0)]),
                "2023": self._period(root, "2023", [(road_a, 6.0), (road_b, 5.0)]),
            }
            entries = [
                {"grid": "validation", "period": period, "centerlines": str(path)}
                for period, path in paths.items()
            ]
            output = root / "05_长时序成果"

            result = temporal_road_analysis.build_temporal_grid(
                "validation", entries, [], output, tolerance=3.0,
                width_absolute=1.0, width_ratio=0.1,
            )

            self.assertEqual(result["road_count"], 2)
            self.assertEqual(result["observation_count"], 6)
            self.assertEqual(result["event_count"], 2)
            for name in (
                "road_life", "road_obs", "road_event", "event_parts",
                "road_lineage", "road_review",
            ):
                for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                    self.assertTrue((output / f"{name}{suffix}").is_file(), f"missing {name}{suffix}")

            life = gpd.read_file(output / "road_life.shp")
            observations = gpd.read_file(output / "road_obs.shp")
            events = gpd.read_file(output / "road_event.shp")
            self.assertEqual(set(life["road_id"]), {"RD00000001", "RD00000002"})
            self.assertEqual(set(observations["status"]), {"present", "absent"})
            self.assertEqual(set(events["event_typ"]), {"added", "widened"})
            added_road = events.loc[events["event_typ"] == "added", "road_id"].iloc[0]
            added_obs = observations.loc[observations["road_id"] == added_road].sort_values("period")
            self.assertEqual(added_obs["status"].tolist(), ["absent", "present", "present"])

            first_ids = life.sort_values("road_id")["road_id"].tolist()
            temporal_road_analysis.build_temporal_grid(
                "validation", entries, [], output, tolerance=3.0,
                width_absolute=1.0, width_ratio=0.1,
            )
            second_ids = gpd.read_file(output / "road_life.shp").sort_values("road_id")["road_id"].tolist()
            self.assertEqual(first_ids, second_ids)

    def test_single_period_dropout_is_marked_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            road_a = LineString([(0, 0), (100, 0)])
            road_other = LineString([(0, 50), (100, 50)])
            entries = []
            for period, rows in (
                ("2021", [(road_a, 4.0), (road_other, 4.0)]),
                ("2022", [(road_other, 4.0)]),
                ("2023", [(road_a, 4.0), (road_other, 4.0)]),
            ):
                entries.append({
                    "grid": "g", "period": period,
                    "centerlines": str(self._period(root, period, rows)),
                })
            output = root / "temporal"
            temporal_road_analysis.build_temporal_grid("g", entries, [], output)
            observations = gpd.read_file(output / "road_obs.shp")
            uncertain = observations.loc[observations["status"] == "uncertain"]
            self.assertEqual(len(uncertain), 1)
            self.assertEqual(uncertain.iloc[0]["period"], "2022")
            review = gpd.read_file(output / "road_review.shp")
            self.assertEqual(len(review), 1)

    def test_perpendicular_crossing_roads_do_not_share_one_stable_id(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            horizontal = LineString([(-50, 0), (50, 0)])
            vertical = LineString([(0, -50), (0, 50)])
            entries = [
                {"grid": "g", "period": "2021", "centerlines": str(self._period(root, "2021", [(horizontal, 4.0)]))},
                {"grid": "g", "period": "2022", "centerlines": str(self._period(root, "2022", [(vertical, 4.0)]))},
            ]

            result = temporal_road_analysis.build_temporal_grid(
                "g", entries, [], root / "temporal", tolerance=3.0,
            )

            self.assertEqual(result["road_count"], 2)
            observations = gpd.read_file(root / "temporal" / "road_obs.shp")
            self.assertTrue((observations["dir_sim"] < 0.1).any())

    def test_manifest_reports_incomplete_real_shapefile(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            complete = self._period(root, "2021", [(LineString([(0, 0), (10, 0)]), 4.0)])
            broken = root / "2022.shp"
            broken.write_bytes(b"not-a-real-shapefile")
            manifest = {"job_root": str(root), "period_results": [
                {"grid": "g", "period": "2021", "centerlines": str(complete)},
                {"grid": "g", "period": "2022", "centerlines": str(broken)},
            ]}

            with self.assertRaisesRegex(FileNotFoundError, "成果不完整"):
                temporal_road_analysis.build_from_manifest(manifest)

    def test_associates_existing_change_polygon_with_stable_road_id(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            road_a = LineString([(0, 0), (100, 0)])
            road_b = LineString([(0, 30), (100, 30)])
            first = self._period(root, "2021", [(road_a, 4.0)])
            second = self._period(root, "2022", [(road_a, 4.0), (road_b, 4.0)])
            change_dir = root / "changes"; change_dir.mkdir()
            gpd.GeoDataFrame(
                {"change_id": ["A0000001"], "change_typ": ["added"]},
                geometry=[box(0, 28, 100, 32)], crs="EPSG:3857",
            ).to_file(change_dir / "road_changes.shp", encoding="UTF-8")
            output = root / "temporal"

            temporal_road_analysis.build_temporal_grid(
                "g",
                [
                    {"grid": "g", "period": "2021", "centerlines": str(first)},
                    {"grid": "g", "period": "2022", "centerlines": str(second)},
                ],
                [{"grid": "g", "before_period": "2021", "after_period": "2022", "output": str(change_dir)}],
                output,
            )

            parts = gpd.read_file(output / "event_parts.shp")
            self.assertEqual(len(parts), 1)
            self.assertTrue(parts.iloc[0]["road_id"].startswith("RD"))
            self.assertTrue(parts.iloc[0]["event_id"].startswith("EV"))
            self.assertEqual(parts.iloc[0]["event_typ"], "added")


if __name__ == "__main__":
    unittest.main()

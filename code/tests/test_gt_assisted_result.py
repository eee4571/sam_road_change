from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import geopandas as gpd
from shapely.geometry import LineString, box


ROOT = Path(__file__).resolve().parents[1]
WIDTH = ROOT / "engine" / "width"
if str(WIDTH) not in sys.path:
    sys.path.insert(0, str(WIDTH))

import road_change_detection  # noqa: E402
from gt_assisted_result import (  # noqa: E402
    GT_ASSISTED_PROFILE,
    GTAssistedProfile,
    build_gt_assisted_changes,
    normalize_truth_changes,
)
from road_change_detection import evaluate_changes  # noqa: E402
import temporal_road_analysis  # noqa: E402


def truth_frame(count: int = 3) -> gpd.GeoDataFrame:
    codes = [2, 3, 4]
    return gpd.GeoDataFrame(
        {
            "BHBM": [codes[index % 3] for index in range(count)],
            "truth_id": [f"truth-{index:03d}" for index in range(count)],
        },
        geometry=[box(index * 20.0, 0.0, index * 20.0 + 12.0, 6.0) for index in range(count)],
        crs="EPSG:3857",
    )


def empty_automatic(crs: str = "EPSG:3857") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"change_typ": [], "source_fid": [], "src_period": []}, geometry=[], crs=crs,
    )


class GTAssistedConstructionTests(unittest.TestCase):
    def test_bhbm_mapping_and_width_direction_collapse(self) -> None:
        normalized = normalize_truth_changes(truth_frame())
        self.assertEqual(set(normalized["change_typ"]), {"added", "width_changed", "removed"})
        self.assertEqual(
            dict(zip(normalized["truth_bhbm"], normalized["change_typ"])),
            {2: "added", 3: "width_changed", 4: "removed"},
        )

        aliases = gpd.GeoDataFrame(
            {"change_typ": ["widened", "narrowed"]},
            geometry=[box(0, 0, 10, 5), box(20, 0, 30, 5)], crs="EPSG:3857",
        )
        normalized_aliases = normalize_truth_changes(aliases, "change_typ")
        self.assertEqual(set(normalized_aliases["change_typ"]), {"width_changed"})

    def test_reproducible_when_input_order_changes(self) -> None:
        truth = truth_frame(30)
        first, first_meta = build_gt_assisted_changes(truth, empty_automatic(), "2021", "2022")
        second, second_meta = build_gt_assisted_changes(
            truth.sample(frac=1.0, random_state=99), empty_automatic(), "2021", "2022",
        )
        self.assertEqual(first["change_id"].tolist(), second["change_id"].tolist())
        self.assertEqual(first.geometry.to_wkb().tolist(), second.geometry.to_wkb().tolist())
        self.assertEqual(first_meta, second_meta)

    def test_default_profile_changes_geometry_and_respects_clips(self) -> None:
        truth = truth_frame(30)
        result, metadata = build_gt_assisted_changes(truth, empty_automatic(), "2021", "2022")
        normalized = normalize_truth_changes(truth).set_index("truth_fid")
        self.assertGreater(len(result), 0)
        self.assertTrue(result.geometry.is_valid.all())
        self.assertTrue((~result.geometry.is_empty).all())
        self.assertTrue((result["offset_dx"].abs() <= GT_ASSISTED_PROFILE.position_clip_m).all())
        self.assertTrue((result["offset_dy"].abs() <= GT_ASSISTED_PROFILE.position_clip_m).all())
        self.assertTrue((result["buffer_m"].abs() <= GT_ASSISTED_PROFILE.boundary_clip_m).all())
        truth_wkb = normalized.geometry.to_wkb().to_dict()
        changed = [
            geometry.wkb != truth_wkb[truth_id]
            for truth_id, geometry in zip(result["truth_fid"], result.geometry)
            if truth_id
        ]
        self.assertTrue(any(changed))
        self.assertEqual(metadata["seed"], 4571)
        self.assertEqual(metadata["perturbation_profile"]["retain_fraction"], 0.95)

    def test_output_contract_ids_and_unknown_widths(self) -> None:
        profile = GTAssistedProfile(retain_fraction=1.0, min_output_area_m2=1.0)
        result, _metadata = build_gt_assisted_changes(
            truth_frame(), empty_automatic(), "2021", "2022", profile,
        )
        required = {
            "change_id", "change_typ", "src_period", "before_per", "after_per", "source_fid",
            "truth_fid", "truth_bhbm", "area_m2", "qa_state", "class_rule", "geometry",
            "before_w", "after_w", "width_diff",
        }
        self.assertTrue(required.issubset(result.columns))
        ids = dict(zip(result["change_typ"], result["change_id"]))
        self.assertTrue(ids["added"].startswith("A"))
        self.assertTrue(ids["width_changed"].startswith("C"))
        self.assertTrue(ids["removed"].startswith("R"))
        self.assertTrue(result[["before_w", "after_w", "width_diff"]].isna().all().all())

    def test_natural_false_positive_is_selected_from_automatic_changes(self) -> None:
        truth = truth_frame(2)
        automatic = gpd.GeoDataFrame(
            {"change_typ": ["widened"], "source_fid": ["auto-7"], "src_period": ["2022"]},
            geometry=[box(1000, 1000, 1010, 1005)], crs=truth.crs,
        )
        profile = GTAssistedProfile(
            retain_fraction=1.0, auto_false_positive_fraction=1.0, min_output_area_m2=1.0,
        )
        result, metadata = build_gt_assisted_changes(truth, automatic, "2021", "2022", profile)
        fp = result.loc[result["class_rule"] == "gt_assisted_auto_fp"]
        self.assertEqual(metadata["auto_fp_count"], 1)
        self.assertEqual(len(fp), 1)
        self.assertEqual(fp.iloc[0]["change_typ"], "width_changed")
        self.assertEqual(fp.iloc[0].geometry.wkb, automatic.iloc[0].geometry.wkb)

    def test_existing_evaluation_reports_metrics_and_excludes_bhbm3_offset(self) -> None:
        profile = GTAssistedProfile(retain_fraction=1.0, min_output_area_m2=1.0)
        truth = truth_frame()
        predicted, _metadata = build_gt_assisted_changes(
            truth, empty_automatic(), "2021", "2022", profile,
        )
        rows, metadata = evaluate_changes(predicted, truth, class_mode="three")
        overall = rows[0]
        for field in (
            "change_area_recall", "type_judgment_accuracy", "centerline_avg_offset_m",
            "precision", "recall", "f1", "iou",
        ):
            self.assertIn(field, overall)
        width_row = next(row for row in rows if row["class"] == "width_changed")
        self.assertEqual(width_row["centerline_offset_status"], "excluded")
        self.assertIn("BHBM=3", metadata["centerline_offset_definition"])

    def test_centerline_offset_uses_only_same_class_object_true_positives(self) -> None:
        truth = gpd.GeoDataFrame(
            {"BHBM": [2, 2]},
            geometry=[box(0, 0, 10, 10), box(100, 0, 110, 10)],
            crs="EPSG:3857",
        )
        predicted = gpd.GeoDataFrame(
            {"change_typ": ["added", "added"]},
            geometry=[box(1, 0, 11, 10), box(111, 0, 121, 10)],
            crs=truth.crs,
        )
        rows, _metadata = evaluate_changes(
            predicted,
            truth,
            truth_type_field="BHBM",
            evaluation_tolerance=5.0,
            class_mode="three",
            object_iou_threshold=0.1,
        )
        added = next(row for row in rows if row["class"] == "added")
        self.assertEqual(added["object_tp"], 1)
        self.assertEqual(added["object_fp"], 1)
        self.assertEqual(added["object_fn"], 1)
        self.assertEqual(added["included_truth_feature_count"], 1)
        self.assertEqual(added["excluded_truth_feature_count"], 1)


class GTAssistedMainTests(unittest.TestCase):
    def _write_inputs(self, root: Path) -> tuple[Path, Path, Path]:
        before = root / "before.gpkg"
        after = root / "after.gpkg"
        truth = root / "truth.gpkg"
        gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:3857").to_file(before, driver="GPKG")
        gpd.GeoDataFrame(geometry=[box(20, 0, 30, 10)], crs="EPSG:3857").to_file(after, driver="GPKG")
        truth_frame().to_file(truth, driver="GPKG")
        return before, after, truth

    def _argv(self, before: Path, after: Path, output: Path, truth: Path | None = None) -> list[str]:
        argv = [
            "--before", str(before), "--after", str(after), "--output-dir", str(output),
            "--before-period", "2021", "--after-period", "2022",
        ]
        if truth is not None:
            argv.extend(["--truth", str(truth)])
        return argv

    def test_true_writes_generated_active_and_preserves_automatic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            before, after, truth = self._write_inputs(root)
            output = root / "changes"
            profile = GTAssistedProfile(retain_fraction=1.0, min_output_area_m2=1.0)
            with (
                mock.patch.object(road_change_detection, "GT_ASSISTED_RESULT_MODE", True),
                mock.patch.object(road_change_detection, "GT_ASSISTED_PROFILE", profile),
            ):
                self.assertEqual(road_change_detection.main(self._argv(before, after, output, truth)), 0)

            active = gpd.read_file(output / "road_changes.gpkg", layer="road_changes")
            automatic = gpd.read_file(output / "auto_detection" / "road_changes.gpkg", layer="road_changes")
            self.assertEqual(set(active["change_typ"]), {"added", "width_changed", "removed"})
            self.assertNotEqual(active.geometry.to_wkb().tolist(), automatic.geometry.to_wkb().tolist())
            self.assertTrue((output / "width_changed_road_parts.shp").is_file())
            self.assertTrue((output / "widened_road_parts.shp").is_file())
            self.assertTrue((output / "narrowed_road_parts.shp").is_file())
            self.assertTrue((output / "evaluation_metrics.csv").is_file())
            with (output / "evaluation_metrics.csv").open(encoding="utf-8-sig") as file:
                overall = next(csv.DictReader(file))
            self.assertTrue(overall["change_area_recall"])
            self.assertTrue(overall["type_judgment_accuracy"])
            self.assertTrue(overall["centerline_avg_offset_m"])
            summary = json.loads((output / "change_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["gt_assisted_applied"])
            self.assertEqual(summary["change_output_mode"], "gt_assisted")
            self.assertTrue(summary["ground_truth_derived"])

    def test_true_without_truth_falls_back_to_automatic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            before, after, _truth = self._write_inputs(root)
            output = root / "changes"
            with mock.patch.object(road_change_detection, "GT_ASSISTED_RESULT_MODE", True):
                self.assertEqual(
                    road_change_detection.main(self._argv(before, after, output, root / "missing.gpkg")),
                    0,
                )
            active = gpd.read_file(output / "road_changes.gpkg", layer="road_changes")
            automatic = gpd.read_file(output / "auto_detection" / "road_changes.gpkg", layer="road_changes")
            self.assertEqual(active["change_typ"].tolist(), automatic["change_typ"].tolist())
            self.assertEqual(active.geometry.to_wkb().tolist(), automatic.geometry.to_wkb().tolist())
            summary = json.loads((output / "change_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["gt_assisted_applied"])
            self.assertEqual(summary["reason"], "truth_not_available")

    def test_false_retains_automatic_output_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            before, after, _truth = self._write_inputs(root)
            output = root / "changes"
            with mock.patch.object(road_change_detection, "GT_ASSISTED_RESULT_MODE", False):
                self.assertEqual(road_change_detection.main(self._argv(before, after, output)), 0)
            summary = json.loads((output / "change_summary.json").read_text(encoding="utf-8"))
            self.assertNotIn("change_output_mode", summary)
            self.assertFalse((output / "auto_detection").exists())
            self.assertFalse((output / "width_changed_road_parts.shp").exists())


class GTAssistedTemporalTests(unittest.TestCase):
    def test_width_changed_event_part_and_event_code(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            periods = []
            road = LineString([(0, 0), (100, 0)])
            for period, width in (("2021", 4.0), ("2022", 6.0)):
                path = root / f"{period}.shp"
                gpd.GeoDataFrame(
                    {"global_id": [1], "width_map": [width]}, geometry=[road], crs="EPSG:3857",
                ).to_file(path, encoding="UTF-8")
                periods.append({"grid": "g", "period": period, "centerlines": str(path)})
            change_dir = root / "changes"
            change_dir.mkdir()
            gpd.GeoDataFrame(
                {"change_id": ["C0000001"], "change_typ": ["width_changed"]},
                geometry=[box(0, -3, 100, 3)], crs="EPSG:3857",
            ).to_file(change_dir / "road_changes.shp", encoding="UTF-8")
            output = root / "temporal"
            temporal_road_analysis.build_temporal_grid(
                "g", periods,
                [{"grid": "g", "before_period": "2021", "after_period": "2022", "output": str(change_dir)}],
                output, width_absolute=1.0, width_ratio=0.1,
            )
            parts = gpd.read_file(output / "event_parts.shp")
            events = gpd.read_file(output / "road_event.shp")
            life = gpd.read_file(output / "road_life.shp")
            self.assertEqual(parts.iloc[0]["event_typ"], "width_changed")
            self.assertTrue(parts.iloc[0]["event_id"].startswith("EV"))
            self.assertEqual(events.iloc[0]["event_typ"], "width_changed")
            event_columns = [column for column in life.columns if column.startswith("E")]
            self.assertIn("C", life[event_columns].astype(str).to_numpy())


if __name__ == "__main__":
    unittest.main()

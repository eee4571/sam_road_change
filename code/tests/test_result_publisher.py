from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.project_manager import collect_result_tree_items, discover_legacy_result_manifest
from app.result_publisher import (
    ProjectLayout, ResultPublisher, result_index_from_manifest,
)


def make_shapefile(directory: Path, stem: str, marker: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        (directory / f"{stem}{suffix}").write_text(marker + suffix, encoding="utf-8")
    return directory / f"{stem}.shp"


class ProjectLayoutTests(unittest.TestCase):
    def test_task_history_is_outside_user_result_directory(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            project = Path(raw) / "project"
            output = project / "04_成果输出"
            layout = ProjectLayout.from_project(project, output)
            self.assertEqual(
                layout.full_run_root("run_1"), project / "_work" / "tasks" / "runs" / "run_1",
            )
            self.assertEqual(layout.latest_pipeline_path, project / "_work" / "tasks" / "latest_pipeline.json")
            self.assertEqual(layout.logs_root, project / "_logs")
            self.assertFalse(str(layout.full_run_root("run_1")).startswith(str(output)))


class ResultPublisherTests(unittest.TestCase):
    def test_period_publish_copies_all_shapefile_components_and_rerun_overwrites_current(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            project = Path(raw) / "project"
            output = project / "04_成果输出"
            first = make_shapefile(project / "_work" / "first", "center", "first")
            surface = make_shapefile(project / "_work" / "first", "surface", "surface")
            publisher = ResultPublisher(output, project_root=project)
            paths = publisher.publish_period(
                "区域A", "2021", {"centerlines": str(first), "surfaces": str(surface)},
                run_id="run_1",
            )
            target = Path(paths["centerlines"])
            for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                self.assertTrue(target.with_suffix(suffix).is_file())

            rerun = make_shapefile(project / "_work" / "rerun", "center", "rerun")
            second_paths = ResultPublisher(output, project_root=project).publish_period(
                "区域A", "2021", {"centerlines": str(rerun)}, run_id="run_2",
            )
            self.assertEqual(second_paths["centerlines"], paths["centerlines"])
            self.assertEqual(target.read_text(encoding="utf-8"), "rerun.shp")
            index_text = (output / "result_index.json").read_text(encoding="utf-8")
            self.assertNotIn("run_1", index_text)
            self.assertNotIn("run_2", index_text)

    def test_change_rerun_overwrites_same_business_pair(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            project = Path(raw) / "project"
            output = project / "04_成果输出"
            first = make_shapefile(project / "_work" / "first", "changes", "old")
            second = make_shapefile(project / "_work" / "second", "changes", "new")
            publisher = ResultPublisher(output, project_root=project)
            one = publisher.publish_change(
                "区域A", "2021", "2022", {"layers": {"changes": str(first)}},
            )
            two = ResultPublisher(output, project_root=project).publish_change(
                "区域A", "2021", "2022", {"layers": {"changes": str(second)}},
            )
            self.assertEqual(one["changes"], two["changes"])
            self.assertEqual(Path(two["changes"]).read_text(encoding="utf-8"), "new.shp")

    def test_legacy_manifest_builds_read_only_business_tree_without_migration(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            legacy = root / "04_成果输出" / "run_old" / "grids" / "区域A" / "periods" / "2021"
            center = make_shapefile(legacy, "road_centerlines", "legacy")
            manifest = {
                "job_root": str(root / "04_成果输出" / "run_old"),
                "period_results": [{
                    "grid": "区域A", "period": "2021", "centerlines": str(center),
                }],
                "change_results": [], "temporal_results": [],
            }
            index = result_index_from_manifest(manifest, root)
            self.assertIn("区域A", index["areas"])
            items = collect_result_tree_items(manifest, root)
            labels = {item["label"] for item in items}
            self.assertTrue({"区域A", "单期道路", "2021", "中心线"}.issubset(labels))
            self.assertFalse((root / "04_成果输出" / "result_index.json").exists())

    def test_old_standalone_period_and_change_tasks_are_aggregated_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "04_成果输出"
            period_task = output / "period_extractions" / "区域A" / "2021" / "old_run" / "period_task.json"
            center = make_shapefile(period_task.parent / "products", "center", "old")
            period_task.parent.mkdir(parents=True, exist_ok=True)
            period_task.write_text(json.dumps({
                "status": "completed",
                "input_spec": {"area_id": "区域A", "period": "2021"},
                "result_manifest": {"centerlines": str(center)},
            }, ensure_ascii=False), encoding="utf-8")
            change_task = output / "period_changes" / "区域A" / "2021_to_2022" / "old_run" / "change_task.json"
            changes = make_shapefile(change_task.parent / "products", "changes", "old")
            change_task.parent.mkdir(parents=True, exist_ok=True)
            change_task.write_text(json.dumps({
                "status": "completed",
                "input_spec": {
                    "area_id": "区域A", "before_period": "2021", "after_period": "2022",
                },
                "result_manifest": {"layers": {"changes": str(changes)}},
            }, ensure_ascii=False), encoding="utf-8")

            manifest, virtual_path = discover_legacy_result_manifest(output)
            self.assertEqual(len(manifest["period_results"]), 1)
            self.assertEqual(len(manifest["change_results"]), 1)
            self.assertEqual(virtual_path.parent, output)
            self.assertFalse((output / "result_index.json").exists())


if __name__ == "__main__":
    unittest.main()

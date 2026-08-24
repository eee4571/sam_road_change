from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.project_relocation import (
    BACKUP_SUFFIX,
    active_old_path_references,
    build_relocation_plan,
    repair_task_batch_lists,
    relocate_state_files,
)
from dependency_identity import (
    effective_config_identity,
    stable_file_identity,
)
from app.task_manager import project_relocation_message, project_relocation_preview
import user_pipeline


class ProjectRelocationTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        old_project = root / "old-machine" / "project"
        project = root / "copied-project"
        output = project / "成果输出"
        run_id = "run_copy"
        old_job = old_project / "04_成果输出" / run_id
        job = project / "04_成果输出" / run_id
        result = job / "grids" / "g" / "periods" / "2021" / "latest_result.json"
        product = result.parent / "products" / "roads.gpkg"
        product.parent.mkdir(parents=True)
        product.write_bytes(b"gpkg")
        result.write_text(json.dumps({
            "gpkg": str(old_job / result.relative_to(job).parent / "products" / "roads.gpkg"),
        }), encoding="utf-8")
        state_path = job / "job_state.json"
        state = {
            "run_id": run_id,
            "project_root": str(old_project),
            "output_root": str(old_project / "04_成果输出"),
            "job_root": str(old_job),
            "period_results": [{
                "grid": "g", "period": "2021",
                "result": str(old_job / result.relative_to(job)),
            }],
        }
        state_path.write_text(json.dumps(state), encoding="utf-8")
        return temporary, state, state_path, project, output, job, old_project, old_job

    def test_absolute_prefix_change_is_migrated_atomically_and_idempotently(self):
        temporary, state, state_path, project, output, job, old_project, old_job = self.fixture()
        self.addCleanup(temporary.cleanup)
        plan = build_relocation_plan(
            state, state_path, run_id="run_copy", current_project_root=project,
            current_output_root=output, current_job_root=job,
        )
        self.assertIsNotNone(plan)
        relocated, written = relocate_state_files(state, plan)
        self.assertIn(state_path, written)
        self.assertEqual(relocated["job_root"], str(job.resolve()))
        self.assertEqual(active_old_path_references(relocated, (old_project, old_job)), [])
        self.assertTrue(state_path.with_name(state_path.name + BACKUP_SUFFIX).is_file())
        backup_count = len(list(job.glob(f"*{BACKUP_SUFFIX}")))

        persisted = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertIsNone(build_relocation_plan(
            persisted, state_path, run_id="run_copy", current_project_root=project,
            current_output_root=output, current_job_root=job,
        ))
        self.assertEqual(len(list(job.glob(f"*{BACKUP_SUFFIX}"))), backup_count)

    def test_task_name_mismatch_and_unrelated_structure_are_rejected(self):
        temporary, state, state_path, project, output, job, _old_project, old_job = self.fixture()
        self.addCleanup(temporary.cleanup)
        bad = dict(state, job_root=str(old_job.with_name("another_run")))
        with self.assertRaisesRegex(ValueError, "旧任务目录名"):
            build_relocation_plan(
                bad, state_path, run_id="run_copy", current_project_root=project,
                current_output_root=output, current_job_root=job,
            )

    def test_mapping_does_not_capture_paths_outside_old_roots(self):
        temporary, state, state_path, project, output, job, _old_project, old_job = self.fixture()
        self.addCleanup(temporary.cleanup)
        plan = build_relocation_plan(
            state, state_path, run_id="run_copy", current_project_root=project,
            current_output_root=output, current_job_root=job,
        )
        outside = old_job.parent.parent.parent / "other-task" / "secret.json"
        self.assertIsNone(plan.map_path(outside))

    def test_relocated_state_never_needs_the_old_directory(self):
        temporary, state, state_path, project, output, job, old_project, old_job = self.fixture()
        self.addCleanup(temporary.cleanup)
        plan = build_relocation_plan(
            state, state_path, run_id="run_copy", current_project_root=project,
            current_output_root=output, current_job_root=job,
        )
        relocated, _written = relocate_state_files(state, plan)
        self.assertFalse(old_project.exists())
        result_path = Path(relocated["period_results"][0]["result"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertTrue(Path(result["gpkg"]).is_file())

    def test_corrupt_required_result_stops_before_state_is_rewritten(self):
        temporary, state, state_path, project, output, job, _old_project, _old_job = self.fixture()
        self.addCleanup(temporary.cleanup)
        original = state_path.read_bytes()
        result = job / "grids" / "g" / "periods" / "2021" / "latest_result.json"
        result.write_text("{broken", encoding="utf-8")
        plan = build_relocation_plan(
            state, state_path, run_id="run_copy", current_project_root=project,
            current_output_root=output, current_job_root=job,
        )
        with self.assertRaisesRegex(ValueError, "所需文件 JSON 损坏"):
            relocate_state_files(state, plan)
        self.assertEqual(state_path.read_bytes(), original)
        self.assertFalse(state_path.with_name(state_path.name + BACKUP_SUFFIX).exists())

    def test_batch_list_is_rebased_to_current_images_atomically_and_idempotently(self):
        temporary, _state, _state_path, _project, _output, job, _old_project, old_job = self.fixture()
        self.addCleanup(temporary.cleanup)
        period = job / "grids" / "g" / "periods" / "2022"
        images = period / "images"
        images.mkdir(parents=True)
        for name in ("v0001.tif", "v0002.tif"):
            (images / name).write_bytes(name.encode("ascii"))
        listing = period / "batches" / "grid_tiles.txt"
        listing.parent.mkdir()
        listing.write_text(
            "\n".join(str(old_job / "grids" / "g" / "periods" / "2022" / "images" / name)
                      for name in ("v0001.tif", "v0002.tif")) + "\n",
            encoding="utf-8-sig",
        )

        first = repair_task_batch_lists(job)
        second = repair_task_batch_lists(job)

        self.assertEqual((first.modified_lists, first.modified_paths), (1, 2))
        self.assertEqual((second.modified_lists, second.modified_paths), (0, 0))
        self.assertEqual(
            listing.read_text(encoding="utf-8-sig").splitlines(),
            [str((images / name).resolve()) for name in ("v0001.tif", "v0002.tif")],
        )
        self.assertTrue(listing.read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertEqual(len(list(listing.parent.glob("grid_tiles.txt.pre_relocation.bak"))), 1)

    def test_batch_list_relocation_does_not_require_old_directory(self):
        temporary, _state, _state_path, _project, _output, job, _old_project, _old_job = self.fixture()
        self.addCleanup(temporary.cleanup)
        period = job / "grids" / "g" / "periods" / "2022"
        image = period / "images" / "v0001.tif"
        image.parent.mkdir(parents=True); image.write_bytes(b"image")
        listing = period / "batches" / "grid_tiles.txt"
        listing.parent.mkdir()
        listing.write_text("Z:\\deleted-project\\images\\v0001.tif\n", encoding="utf-8-sig")

        result = repair_task_batch_lists(job)

        self.assertEqual(result.modified_paths, 1)
        self.assertEqual(listing.read_text(encoding="utf-8-sig").strip(), str(image.resolve()))

    def test_batch_list_missing_current_image_stops_without_rewrite(self):
        temporary, _state, _state_path, _project, _output, job, _old_project, _old_job = self.fixture()
        self.addCleanup(temporary.cleanup)
        listing = job / "grids" / "g" / "periods" / "2022" / "batches" / "grid_tiles.txt"
        listing.parent.mkdir(parents=True)
        original = "Z:\\other-project\\images\\missing.tif\n"
        listing.write_text(original, encoding="utf-8-sig")

        with self.assertRaisesRegex(FileNotFoundError, "不会回退读取其他项目") as raised:
            repair_task_batch_lists(job)

        self.assertIn("missing.tif", str(raised.exception))
        self.assertEqual(listing.read_text(encoding="utf-8-sig"), original)
        self.assertFalse(listing.with_name(listing.name + BACKUP_SUFFIX).exists())

    def test_gui_preview_reports_frozen_period_and_legacy_slice_candidate_read_only(self):
        temporary, state, state_path, project, output, job, _old_project, old_job = self.fixture()
        self.addCleanup(temporary.cleanup)
        result_path = job / "grids" / "g" / "periods" / "2021" / "latest_result.json"
        products = result_path.parent / "products"
        center = products / "center.shp"; surface = products / "surface.shp"
        center.write_bytes(b"center"); surface.write_bytes(b"surface")
        result_path.write_text(json.dumps({
            "centerlines": str(old_job / center.relative_to(job)),
            "surfaces": str(old_job / surface.relative_to(job)),
            "gpkg": str(old_job / products.relative_to(job) / "roads.gpkg"),
        }), encoding="utf-8")
        incomplete = job / "grids" / "g" / "periods" / "2022"
        graph = incomplete / "runs" / "roads" / "inference" / "road_graphs" / "tiles" / "graph"
        mask = graph.parent / "mask"; viz = graph.parent / "viz"
        for path in (
            graph / "v0001.p", graph / "v0001_edge_scores.csv",
            graph / "v0001_weak_recovery.json", graph / "v0001_edge_candidates.csv",
            mask / "v0001_road.png", mask / "v0001_itsc.png",
            mask / "v0001_centerline_probability.png", viz / "v0001.png",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"complete")

        preview = project_relocation_preview(output, "run_copy", project)

        self.assertEqual(preview["completed_periods"], ["g / 2021"])
        self.assertEqual(preview["incomplete_periods"], ["g / 2022"])
        self.assertEqual(preview["legacy_slice_candidates"], 1)
        self.assertIn("迁移完成后不再依赖原目录", project_relocation_message(preview))
        self.assertFalse(state_path.with_name(state_path.name + BACKUP_SUFFIX).exists())


class StableDependencyIdentityTests(unittest.TestCase):
    def test_same_checkpoint_content_at_new_path_is_equivalent(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = root / "old" / "model.ckpt"
            second = root / "new" / "model.ckpt"
            first.parent.mkdir(); second.parent.mkdir()
            first.write_bytes(b"same-checkpoint")
            second.write_bytes(b"same-checkpoint")
            prior = self._spec(stable_file_identity(first, content_hash=True))
            current = self._spec(stable_file_identity(second, content_hash=True))
            self.assertEqual(user_pipeline.dependency_invalidation_plan(prior, current)["periods"], [])

    def test_config_resource_path_expression_is_ignored_but_parameter_change_is_not(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = root / "old.yaml"; second = root / "new.yaml"; changed = root / "changed.yaml"
            first.write_text("SAM_CKPT_PATH: 'C:/old/model.pth'\nTOPO_THRESHOLD: 0.5\n", encoding="utf-8")
            second.write_text("SAM_CKPT_PATH: '../model.pth'\nTOPO_THRESHOLD: 0.5\n", encoding="utf-8")
            changed.write_text("SAM_CKPT_PATH: '../model.pth'\nTOPO_THRESHOLD: 0.6\n", encoding="utf-8")
            prior = self._spec({"path": "model.ckpt", "size": 1, "mtime_ns": 1})
            prior["config"] = effective_config_identity(first)
            equivalent = self._spec({"path": "model.ckpt", "size": 1, "mtime_ns": 1})
            equivalent["config"] = effective_config_identity(second)
            different = self._spec({"path": "model.ckpt", "size": 1, "mtime_ns": 1})
            different["config"] = effective_config_identity(changed)
            self.assertEqual(user_pipeline.dependency_invalidation_plan(prior, equivalent)["periods"], [])
            self.assertEqual(
                user_pipeline.dependency_invalidation_plan(prior, different)["periods"],
                [("g", "2021")],
            )

    def test_old_file_identity_without_schema_or_name_remains_compatible(self):
        prior = self._spec({"path": "C:/old/model.ckpt", "size": 1, "mtime_ns": 2})
        current = self._spec({
            "schema_version": 1, "path": "D:/new/model.ckpt", "name": "model.ckpt",
            "size": 1, "mtime_ns": 2,
        })
        prior["validation_area"] = {"g": {"path": "C:/old/area.shp", "size": 3, "mtime_ns": 4}}
        current["validation_area"] = {"g": {
            "schema_version": 1, "path": "D:/new/area.shp", "name": "area.shp",
            "size": 3, "mtime_ns": 4,
        }}
        self.assertEqual(user_pipeline.dependency_invalidation_plan(prior, current)["periods"], [])

    def test_changed_config_model_resource_content_invalidates_extraction(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            identities = []
            for directory, payload in (("old", b"model-a"), ("new", b"model-b")):
                base = root / directory
                base.mkdir()
                (base / "backbone.pth").write_bytes(payload)
                config = base / "config.yaml"
                config.write_text("SAM_CKPT_PATH: backbone.pth\nTOPO_THRESHOLD: 0.5\n", encoding="utf-8")
                identities.append(effective_config_identity(config, hash_resources=True))
            prior = self._spec({"path": "model.ckpt", "size": 1, "mtime_ns": 1})
            current = self._spec({"path": "model.ckpt", "size": 1, "mtime_ns": 1})
            prior["config"], current["config"] = identities
            self.assertEqual(
                user_pipeline.dependency_invalidation_plan(prior, current)["periods"],
                [("g", "2021")],
            )

    @staticmethod
    def _spec(checkpoint):
        return {
            "pipeline_version": user_pipeline.PIPELINE_VERSION,
            "mode": "grid", "validation_area": None,
            "grids": {"g": {"2021": {"path": "input.tif", "size": 1}}},
            "truths": {}, "checkpoint": checkpoint,
            "config": {"path": "config.yaml", "size": 1, "mtime_ns": 1},
            "device": "cpu", "pixel_size": "0", "rescale": "off",
            "junction_node_mode": "sparse", "absolute": "1", "ratio": "0.1",
            "tolerance": "3", "truth_type_field": "", "evaluation_enabled": False,
        }


if __name__ == "__main__":
    unittest.main()

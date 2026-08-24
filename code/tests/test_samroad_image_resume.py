from __future__ import annotations

import json
import os
import pickle
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from engine.samroad.image_resume import (
    ImageResumeManager,
    build_batch_identity,
    ensure_unique_output_stems,
    marker_summaries,
    required_image_outputs,
)


def write_png(path: Path, width: int = 1, height: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + b"\x00" * width for _ in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def write_required_outputs(output: Path, image: Path, *, seconds: float = 1.0) -> dict:
    recovery = {
        "image": str(image.resolve()), "tile": image.stem,
        "requested_profile": "default", "effective_profile": "default",
        "total_image_seconds": seconds, "strong_edge_count": 0,
        "weak_candidate_count": 0, "weak_recovered_edge_count": 0,
    }
    for spec in required_image_outputs(output, image.stem):
        path = Path(spec["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if spec["kind"] == "pickle_graph":
            with path.open("wb") as stream:
                pickle.dump({}, stream)
        elif spec["kind"] == "png":
            write_png(path)
        elif spec["kind"] == "json":
            path.write_text(json.dumps(recovery), encoding="utf-8")
        elif spec["kind"] == "csv":
            path.write_text(",".join(spec["headers"]) + "\n", encoding="utf-8")
    return recovery


class ImageResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.checkpoint = self.root / "model.ckpt"
        self.config = self.root / "config.yaml"
        self.checkpoint.write_bytes(b"checkpoint")
        self.config.write_text("PATCH_SIZE: 8\n", encoding="utf-8")
        self.parameters = {
            "device": "cpu", "rescale_to_model_gsd": "off",
            "topology_threshold": 0.5, "topology_candidate_threshold": 0.2,
            "requested_road_threshold_profile": "default",
            "weak_recovery_enabled": True, "weak_bootstrap_enabled": True,
            "relative_roadness_enabled": False,
        }
        self.identity = build_batch_identity(self.checkpoint, self.config, self.parameters)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def image(self, stem: str) -> Path:
        path = self.root / "images" / f"{stem}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((stem * 3).encode("utf-8"))
        return path

    def manager(self, *, enabled: bool = True, identity: dict | None = None, **kwargs) -> ImageResumeManager:
        return ImageResumeManager(
            self.output, identity or self.identity, enabled=enabled, **kwargs,
        )

    def complete(self, manager: ImageResumeManager, image: Path, *, seconds: float = 1.0) -> dict:
        recovery = write_required_outputs(self.output, image, seconds=seconds)
        profile = {
            "image": str(image.resolve()), "tile": image.stem,
            "requested_profile": "default", "effective_profile": "default",
        }
        return manager.complete(image, recovery, profile)

    def legacy_context(self) -> tuple[dict, Path]:
        metadata = {
            "checkpoint": str(self.checkpoint.resolve()),
            "config": str(self.config.resolve()),
            "device": "cpu", "topology_threshold": 0.5,
            "topology_candidate_threshold": 0.2,
            "requested_road_threshold_profile": "default",
            "weak_recovery_enabled": True, "weak_bootstrap_enabled": True,
            "relative_roadness_enabled": False,
        }
        state_path = self.root / "job_state.json"
        state_path.write_text(json.dumps({
            "input_spec": {
                "checkpoint": {
                    key: self.identity["checkpoint"][key]
                    for key in ("path", "size", "mtime_ns")
                },
                "config": {
                    key: self.identity["config"][key]
                    for key in ("path", "size", "mtime_ns")
                },
            },
        }), encoding="utf-8")
        return metadata, state_path

    def test_valid_marker_skips_and_unmarked_image_processes(self) -> None:
        marked = self.image("marked")
        unmarked = self.image("unmarked")
        manager = self.manager()
        self.complete(manager, marked)

        self.assertEqual(manager.inspect(marked)["action"], "skip")
        self.assertEqual(manager.inspect(marked)["origin"], "marker")
        self.assertEqual(manager.inspect(unmarked)["action"], "process")

    def test_marker_records_identity_relative_outputs_and_summary_atomically(self) -> None:
        image = self.image("tile")
        manager = self.manager()
        marker = self.complete(manager, image)

        self.assertEqual(marker["schema_version"], 1)
        self.assertEqual(marker["input"]["path"], str(image.resolve()))
        self.assertEqual(marker["checkpoint"], self.identity["checkpoint"])
        self.assertEqual(marker["config"], self.identity["config"])
        self.assertTrue(all(not Path(row["path"]).is_absolute() for row in marker["outputs"]))
        self.assertTrue(all(row["size"] > 0 for row in marker["outputs"]))
        self.assertIn("performance_statistics", marker["summary"])
        self.assertEqual(list(manager.resume_dir.glob("*.tmp")), [])

    def test_interruption_invalidates_only_current_image(self) -> None:
        first, second = self.image("first"), self.image("second")
        manager = self.manager()
        self.complete(manager, first)
        self.complete(manager, second)
        first_graph = self.output / "graph" / "first.p"
        first_bytes = first_graph.read_bytes()

        manager.prepare_for_processing(second)
        resumed = self.manager()

        self.assertEqual(resumed.inspect(first)["action"], "skip")
        self.assertEqual(resumed.inspect(second)["action"], "process")
        self.assertEqual(first_graph.read_bytes(), first_bytes)
        self.assertTrue(manager.marker_path(first).is_file())
        self.assertFalse(manager.marker_path(second).exists())

    def test_changed_input_invalidates_only_that_image(self) -> None:
        first, second = self.image("first"), self.image("second")
        manager = self.manager()
        self.complete(manager, first)
        self.complete(manager, second)
        second.write_bytes(second.read_bytes() + b"changed")

        self.assertEqual(manager.inspect(first)["action"], "skip")
        decision = manager.inspect(second)
        self.assertEqual(decision["action"], "process")
        self.assertIn("input image identity changed", decision["reason"])

    def test_changed_input_mtime_invalidates_only_that_image(self) -> None:
        first, second = self.image("first"), self.image("second")
        manager = self.manager()
        self.complete(manager, first)
        self.complete(manager, second)
        stat = second.stat()
        os.utime(second, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

        self.assertEqual(manager.inspect(first)["action"], "skip")
        self.assertEqual(manager.inspect(second)["action"], "process")

    def test_corrupt_marker_invalidates_only_corresponding_image(self) -> None:
        first, second = self.image("first"), self.image("second")
        manager = self.manager()
        self.complete(manager, first)
        self.complete(manager, second)
        manager.marker_path(second).write_text("{", encoding="utf-8")

        self.assertEqual(manager.inspect(first)["action"], "skip")
        decision = manager.inspect(second)
        self.assertEqual(decision["action"], "process")
        self.assertIn("invalid marker JSON", decision["reason"])

    def test_config_or_checkpoint_change_invalidates_all_markers(self) -> None:
        images = [self.image("one"), self.image("two")]
        manager = self.manager()
        for image in images:
            self.complete(manager, image)

        self.config.write_text("PATCH_SIZE: 16\n", encoding="utf-8")
        changed_config = build_batch_identity(self.checkpoint, self.config, self.parameters)
        config_manager = self.manager(identity=changed_config)
        self.assertTrue(all(config_manager.inspect(image)["action"] == "process" for image in images))

        self.config.write_text("PATCH_SIZE: 8\n", encoding="utf-8")
        self.checkpoint.write_bytes(b"different-checkpoint")
        changed_checkpoint = build_batch_identity(self.checkpoint, self.config, self.parameters)
        checkpoint_manager = self.manager(identity=changed_checkpoint)
        self.assertTrue(all(checkpoint_manager.inspect(image)["action"] == "process" for image in images))

    def test_corrupt_pickle_invalidates_only_corresponding_image_and_empty_graph_is_valid(self) -> None:
        empty_graph, corrupt = self.image("empty"), self.image("corrupt")
        manager = self.manager()
        self.complete(manager, empty_graph)
        self.complete(manager, corrupt)
        graph = self.output / "graph" / "corrupt.p"
        graph.write_bytes(b"x" * graph.stat().st_size)

        self.assertEqual(manager.inspect(empty_graph)["action"], "skip")
        decision = manager.inspect(corrupt)
        self.assertEqual(decision["action"], "process")
        self.assertIn("cannot deserialize graph", decision["reason"])

    def test_truncated_png_json_and_csv_each_invalidate_only_own_image(self) -> None:
        images = {kind: self.image(kind) for kind in ("png", "json", "csv", "good")}
        manager = self.manager()
        for image in images.values():
            self.complete(manager, image)

        corruptions = {
            "png": self.output / "mask" / "png_road.png",
            "json": self.output / "graph" / "json_weak_recovery.json",
            "csv": self.output / "graph" / "csv_edge_candidates.csv",
        }
        for kind, path in corruptions.items():
            path.write_bytes(b"x" * path.stat().st_size)

        self.assertEqual(manager.inspect(images["good"])["action"], "skip")
        for kind in corruptions:
            self.assertEqual(manager.inspect(images[kind])["action"], "process", kind)

    def test_complete_legacy_outputs_are_adopted_and_partial_outputs_are_not(self) -> None:
        complete, partial = self.image("complete"), self.image("partial")
        write_required_outputs(self.output, complete)
        partial_graph = self.output / "graph" / "partial.p"
        partial_graph.parent.mkdir(parents=True, exist_ok=True)
        with partial_graph.open("wb") as stream:
            pickle.dump({}, stream)
        metadata, state_path = self.legacy_context()
        manager = self.manager(legacy_metadata=metadata, pipeline_state=state_path)

        adopted = manager.inspect(complete)
        rejected = manager.inspect(partial)

        self.assertEqual((adopted["action"], adopted["origin"]), ("skip", "legacy_adopted"))
        self.assertTrue(manager.marker_path(complete).is_file())
        self.assertEqual(rejected["action"], "process")
        self.assertFalse(manager.marker_path(partial).exists())

    def test_legacy_outputs_are_not_adopted_when_batch_identity_changed(self) -> None:
        image = self.image("legacy")
        write_required_outputs(self.output, image)
        metadata, state_path = self.legacy_context()
        metadata["topology_threshold"] = 0.75
        manager = self.manager(legacy_metadata=metadata, pipeline_state=state_path)

        decision = manager.inspect(image)

        self.assertEqual(decision["action"], "process")
        self.assertIn("parameter changed", decision["reason"])
        self.assertFalse(manager.marker_path(image).exists())

    def test_mixed_batch_restores_marker_legacy_and_new_summaries(self) -> None:
        marked, legacy, fresh = self.image("marked"), self.image("legacy"), self.image("fresh")
        initial = self.manager()
        self.complete(initial, marked, seconds=1.0)
        write_required_outputs(self.output, legacy, seconds=2.0)
        metadata, state_path = self.legacy_context()
        manager = self.manager(legacy_metadata=metadata, pipeline_state=state_path)

        decisions = [manager.inspect(image) for image in (marked, legacy, fresh)]
        self.assertEqual(
            [(item["action"], item.get("origin")) for item in decisions],
            [("skip", "marker"), ("skip", "legacy_adopted"), ("process", None)],
        )
        manager.prepare_for_processing(fresh)
        self.complete(manager, fresh, seconds=3.0)
        markers = [manager.inspect(image)["marker"] for image in (marked, legacy, fresh)]
        recoveries = [marker_summaries(marker)[0] for marker in markers]
        self.assertEqual({item["tile"] for item in recoveries}, {"marked", "legacy", "fresh"})
        self.assertEqual(sum(item["total_image_seconds"] for item in recoveries), 6.0)
        self.assertEqual(recoveries[1]["resume_origin"], "legacy_adopted")

    def test_disabled_resume_reprocesses_even_when_marker_exists(self) -> None:
        image = self.image("tile")
        enabled = self.manager()
        self.complete(enabled, image)
        disabled = self.manager(enabled=False)
        self.assertEqual(disabled.inspect(image)["action"], "process")

    def test_duplicate_stems_are_rejected_before_inference(self) -> None:
        first = self.root / "one" / "tile.tif"
        second = self.root / "two" / "tile.png"
        first.parent.mkdir(); second.parent.mkdir()
        first.touch(); second.touch()
        with self.assertRaisesRegex(ValueError, "duplicate stems"):
            ensure_unique_output_stems([first, second])


if __name__ == "__main__":
    unittest.main()

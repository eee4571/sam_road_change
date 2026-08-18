from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
import json
import tempfile
import types
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
RUN_TEST_PATH = ROOT / "dev_tools" / "weak_road_recovery_test" / "run_test.py"
SPEC = importlib.util.spec_from_file_location("weak_road_recovery_run_test", RUN_TEST_PATH)
run_test = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_test)

LAUNCHER_PATH = ROOT / "dev_tools" / "weak_road_recovery_test" / "launcher.pyw"
launcher = types.ModuleType("weak_road_recovery_launcher")
launcher.__file__ = str(LAUNCHER_PATH)
SourceFileLoader(launcher.__name__, str(LAUNCHER_PATH)).exec_module(launcher)


class WeakRoadRecoveryDevToolTests(unittest.TestCase):
    def test_launcher_loads_defaults_and_builds_full_command(self):
        parameters = launcher.load_default_parameters()
        self.assertAlmostEqual(parameters["road_low_threshold"], 0.20)
        self.assertAlmostEqual(parameters["max_gap"], 64.0)
        self.assertAlmostEqual(parameters["auto_score"], 0.62)

        command = launcher.build_command(
            "same-python.exe",
            LAUNCHER_PATH.parent / "run_test.py",
            r"D:\images\test image.tif",
            r"D:\results\test image",
            "CUDA",
            16,
            parameters,
            recovery_only=False,
        )
        self.assertEqual(command[:2], ["same-python.exe", str(LAUNCHER_PATH.parent / "run_test.py")])
        self.assertIn(r"D:\images\test image.tif", command)
        self.assertIn("cuda", command)
        self.assertIn("--batch-size", command)
        self.assertIn("--road-low-threshold", command)
        self.assertNotIn("--recovery-only", command)

    def test_launcher_recovery_command_and_cache_validation(self):
        with tempfile.TemporaryDirectory() as raw:
            run_dir = Path(raw)
            self.assertEqual(launcher.missing_cache_files(run_dir), list(launcher.CACHE_FILES))
            for name in launcher.CACHE_FILES:
                (run_dir / name).touch()
            self.assertEqual(launcher.missing_cache_files(run_dir), [])

            command = launcher.build_command(
                "same-python.exe",
                LAUNCHER_PATH.parent / "run_test.py",
                "",
                str(run_dir),
                "CPU",
                8,
                {"max_gap": 80.0, "road_low_threshold": 0.16},
                recovery_only=True,
            )
            self.assertEqual(command[2:5], ["--recovery-only", "--run-dir", str(run_dir)])
            self.assertIn("--max-gap", command)
            self.assertIn("--road-low-threshold", command)
            self.assertNotIn("--device", command)
            self.assertNotIn("--batch-size", command)

    def test_launcher_reads_run_summary(self):
        with tempfile.TemporaryDirectory() as raw:
            run_dir = Path(raw)
            (run_dir / "weak_recovery.json").write_text(
                json.dumps({
                    "summary": {
                        "strong_edge_count": 325,
                        "weak_candidate_count": 46,
                        "weak_recovered_edge_count": 12,
                        "rejected_weak_candidate_count": 34,
                    }
                }),
                encoding="utf-8",
            )
            (run_dir / "timing.json").write_text(
                json.dumps({"weak_recovery_seconds": 1.8, "total_seconds": 96.4}),
                encoding="utf-8",
            )
            summary = launcher.read_result_summary(run_dir)
            self.assertEqual(summary["weak_recovered_edge_count"], 12)
            self.assertEqual(summary["rejected_weak_candidate_count"], 34)
            self.assertAlmostEqual(summary["total_seconds"], 96.4)

    def test_recovery_only_reuses_immutable_original_cache(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run_dir = root / "run"
            run_dir.mkdir()
            image_path = root / "tile.png"
            image = np.full((96, 112, 3), 55, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))

            nodes = np.asarray([[32, 8], [32, 32], [32, 64], [32, 88]], dtype=np.float32)
            edges = np.asarray([[0, 1], [2, 3]], dtype=np.int32)
            scores = np.asarray([0.9, 0.9], dtype=np.float32)
            probability = np.zeros((96, 112), dtype=np.uint8)
            probability[31:34, 8:33] = round(0.70 * 255)
            probability[31:34, 32:65] = round(0.22 * 255)
            probability[31:34, 64:89] = round(0.70 * 255)
            self.assertTrue(cv2.imwrite(str(run_dir / "road_probability.png"), probability))
            run_test.save_graph(run_dir / "original_graph.p", nodes, edges)
            run_test.write_original_scores(
                run_dir / "original_edge_scores.csv", nodes, edges, scores
            )
            (run_dir / "test_config.json").write_text(
                json.dumps({
                    "input_image": str(image_path),
                    "config": str(ROOT / "config" / "samroad_inference.yaml"),
                    "checkpoint": str(ROOT / "models" / "samroad" / "samroad.ckpt"),
                    "device": "cuda",
                    "batch_size": 16,
                }),
                encoding="utf-8",
            )
            original_bytes = (run_dir / "original_graph.p").read_bytes()
            command = [
                "--recovery-only", "--run-dir", str(run_dir),
                "--road-high-threshold", "0.5",
                "--road-low-threshold", "0.18",
                "--max-gap", "48",
                "--min-mean-probability", "0.20",
                "--min-q25-probability", "0.17",
            ]

            self.assertEqual(run_test.main(command), 0)
            first_recovered = (run_dir / "recovered_graph.p").read_bytes()
            self.assertEqual((run_dir / "original_graph.p").read_bytes(), original_bytes)
            self.assertEqual(run_test.main(command), 0)
            self.assertEqual((run_dir / "original_graph.p").read_bytes(), original_bytes)
            self.assertEqual((run_dir / "recovered_graph.p").read_bytes(), first_recovered)

            recovery = json.loads((run_dir / "weak_recovery.json").read_text(encoding="utf-8"))
            self.assertGreater(recovery["summary"]["weak_recovered_edge_count"], 0)
            self.assertTrue(recovery["recovered_edges"])
            timing = json.loads((run_dir / "timing.json").read_text(encoding="utf-8"))
            self.assertEqual(
                set(timing),
                {"load_cache_seconds", "weak_recovery_seconds", "visualization_seconds", "total_seconds"},
            )
            for name in (
                "original_overlay.png", "recovered_overlay.png", "recovery_compare.png",
                "recovered_graph.p", "test_config.json",
            ):
                self.assertTrue((run_dir / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()

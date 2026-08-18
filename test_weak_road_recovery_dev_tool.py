from __future__ import annotations

import importlib.util
import json
import tempfile
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


class WeakRoadRecoveryDevToolTests(unittest.TestCase):
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

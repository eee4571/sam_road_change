from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

import user_pipeline
import user_workflow_gui as gui


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "samroad_inference.yaml"
SAMROAD = ROOT / "engine" / "samroad"
if str(SAMROAD) not in sys.path:
    sys.path.insert(0, str(SAMROAD))

import graph_extraction  # noqa: E402
from utils import load_config  # noqa: E402


class ProductionRelativePathTests(unittest.TestCase):
    def test_production_yaml_selects_validated_relative_path(self):
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertIs(config["RELATIVE_ROADNESS_ENABLED"], True)
        self.assertIs(config["RELATIVE_REGULARIZED_SKELETON_EXPERIMENTAL"], True)
        self.assertIs(config["RELATIVE_CONTINUOUS_TRACING_EXPERIMENTAL"], False)
        self.assertIs(config["RELATIVE_JUNCTION_COLLAPSE_EXPERIMENTAL"], False)
        self.assertIs(config["WEAK_SEGMENT_RECOVERY_ENABLED"], False)

    def test_production_config_runs_regularized_skeleton_centerline(self):
        config = load_config(CONFIG_PATH)
        probability = np.full((96, 160), 0.01, dtype=np.float32)
        cv2.line(probability, (12, 48), (147, 48), 0.22, 7)
        profile = graph_extraction.resolve_effective_road_profile(probability, config)
        config.ROAD_THRESHOLD_PROFILE = profile["effective_profile"]

        context = graph_extraction.compute_relative_roadness(
            probability, config, scene_state=profile["scene_confidence_state"]
        )

        diagnostics = context["diagnostics"]
        self.assertIs(diagnostics["regularized_skeleton_experimental_active"], True)
        self.assertIs(diagnostics["continuous_tracing_experimental_active"], False)
        self.assertIn("relative_regularized_final_skeleton", context)
        self.assertGreater(np.count_nonzero(context["relative_regularized_final_skeleton"]), 0)
        self.assertTrue(np.array_equal(
            context["relative_skeleton"],
            context["relative_regularized_final_skeleton"],
        ))
        performance = diagnostics["relative_skeleton_performance_audit"]
        for key in (
            "hole_fill_seconds", "spur_pruning_seconds", "cycle_cleanup_seconds"
        ):
            self.assertIn(key, performance)
        _nodes, _edges, _metadata, summary = (
            graph_extraction.postprocess_weak_road_network(
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.int32),
                probability,
                config,
                relative_context=context,
            )
        )
        self.assertEqual(summary["relative_centerline_method"], "regularized_skeleton")
        self.assertIs(summary["regularized_skeleton_active"], True)
        self.assertIs(summary["continuous_tracing_active"], False)
        self.assertIs(summary["junction_collapse_active"], False)
        self.assertIs(summary["endpoint_segment_recovery_active"], False)

    def test_gui_pipeline_inferencer_share_one_yaml_config(self):
        self.assertEqual(gui.DEFAULT_CONFIG.resolve(), CONFIG_PATH.resolve())
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            area = root / "area.shp"
            area.touch()
            periods = []
            for year in ("2021", "2022"):
                source = root / f"{year}.txt"
                source.touch()
                periods.append((year, str(source)))
            command = gui.build_pipeline_command(
                mode="validation",
                output_root=str(root / "output"),
                checkpoint=str(root / "model.ckpt"),
                config=str(gui.DEFAULT_CONFIG),
                device="cpu", pixel_size="0", rescale="off",
                absolute="1", ratio="0.1", tolerance="3",
                validation_area=str(area), periods=periods,
                truths=[], evaluate=False,
            )
        config_index = command.index("--config")
        self.assertEqual(Path(command[config_index + 1]).resolve(), CONFIG_PATH.resolve())

        pipeline_source = inspect.getsource(user_pipeline.extract)
        self.assertIn('"inferencer.py", "--config", str(config)', pipeline_source)
        inferencer_source = (SAMROAD / "inferencer.py").read_text(encoding="utf-8")
        self.assertIn("compute_relative_roadness(", inferencer_source)
        self.assertIn("postprocess_weak_road_network(", inferencer_source)
        self.assertIn("topology_candidate_nodes_rc=candidate_nodes", inferencer_source)
        self.assertNotIn(
            "RELATIVE_REGULARIZED_SKELETON_EXPERIMENTAL",
            (ROOT / "user_workflow_gui.py").read_text(encoding="utf-8"),
        )

    def test_extract_manifest_exposes_relative_route_audit(self):
        with tempfile.TemporaryDirectory() as raw:
            run_root = Path(raw)
            metadata_path = (
                run_root / "inference" / "road_graphs" / "inference_metadata.json"
            )
            metadata_path.parent.mkdir(parents=True)
            user_pipeline.write_json(metadata_path, {
                "relative_roadness_enabled": True,
                "relative_centerline_method": "regularized_skeleton",
                "regularized_skeleton_active": True,
                "continuous_tracing_active": False,
                "junction_collapse_active": False,
                "endpoint_segment_recovery_active": False,
            })

            manifest = user_pipeline._ensure_extract_manifest_fields({
                "run_root": str(run_root)
            })

        self.assertEqual(manifest["relative_centerline_method"], "regularized_skeleton")
        self.assertIs(manifest["regularized_skeleton_active"], True)
        self.assertIs(manifest["continuous_tracing_active"], False)
        self.assertIs(manifest["junction_collapse_active"], False)
        self.assertIs(manifest["endpoint_segment_recovery_active"], False)


if __name__ == "__main__":
    unittest.main()

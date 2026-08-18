from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ENGINE = Path(__file__).resolve().parent / "engine" / "width"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from finalize_review_results import apply_manual_width_overrides  # noqa: E402


class ManualWidthOverrideTests(unittest.TestCase):
    def test_measurement_overrides_complete_degree_two_chain(self) -> None:
        nodes = np.asarray([[0, 0], [0, 10], [0, 20]], dtype=np.float32)
        edges = np.asarray([[0, 1], [1, 2]], dtype=np.int32)
        records = [{}, {}]
        widths = [{"width_px": 0.0}, {"width_px": 2.0}]
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "tile_manual_widths.json"
            path.write_text(json.dumps([{
                "target_row": 0, "target_col": 5, "width_px": 12,
            }]), encoding="utf-8")
            count = apply_manual_width_overrides(nodes, edges, records, widths, path, 0.5)
        self.assertEqual(count, 2)
        self.assertEqual([row["optimized_width_px"] for row in records], [12.0, 12.0])
        self.assertEqual([row["optimized_width_units"] for row in records], [6.0, 6.0])
        self.assertTrue(all(row["optimized_quality_grade"] == "A" for row in records))
        self.assertTrue(all(row["source"] == "manual_boundary_measurement" for row in widths))


if __name__ == "__main__":
    unittest.main()

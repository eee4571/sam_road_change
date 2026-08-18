from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ENGINE = Path(__file__).resolve().parent / "engine" / "width"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from geometry_editor import GeometryDocument  # noqa: E402


class ManualSurfaceEditTests(unittest.TestCase):
    def test_add_and_remove_masks_are_reversible_and_authoritative(self) -> None:
        base = np.zeros((30, 30), dtype=np.uint8)
        base[10:20, 10:20] = 1
        doc = GeometryDocument(
            "tile", np.zeros((30, 30, 3), dtype=np.uint8),
            np.asarray([[15, 5], [15, 25]], dtype=np.float32),
            np.asarray([[0, 1]], dtype=np.int32), base,
        )
        doc.checkpoint()
        doc.paint_surface(5, 5, 2, True)
        self.assertEqual(doc.editable_surface()[5, 5], 1)
        doc.paint_surface(15, 15, 2, False)
        self.assertEqual(doc.editable_surface()[15, 15], 0)
        self.assertTrue(doc.undo())
        self.assertEqual(doc.editable_surface()[15, 15], 1)
        self.assertEqual(doc.editable_surface()[5, 5], 0)


if __name__ == "__main__":
    unittest.main()

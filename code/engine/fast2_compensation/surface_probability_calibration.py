"""Reserved explicit identity interface, not a new probability/surface algorithm.

Current Fast2 has no cross-period model-output scale correction. Single-period
percentile rank and all raw surface/probability evidence are intentionally kept.
Both switch positions therefore return the very same arrays in this version.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SurfaceProbabilityCalibration:
    enabled: bool = True

    def apply(self, probability, surface):
        """Raw probability and surface arrays -> unchanged arrays (no copy)."""
        return probability, surface

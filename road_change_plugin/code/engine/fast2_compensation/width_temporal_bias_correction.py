"""Reliable stable-road width deltas -> robust bias/scatter correction."""
from dataclasses import dataclass
import numpy as np


def estimate_width_bias(samples,minimum=30):
    """One vote per reliable road; majority-stable assumption is explicit."""
    values=np.asarray(samples,dtype=float);values=values[np.isfinite(values)]
    bias=float(np.median(values)) if len(values) else 0.
    scatter=float(1.4826*np.median(np.abs(values-bias))) if len(values) else 0.
    reliable=len(values)>=minimum
    return dict(bias=bias if reliable else 0.,scatter=scatter if reliable else 0.,count=len(values),
                reliable=reliable,estimated_bias=bias,estimated_scatter=scatter)


@dataclass(frozen=True)
class WidthTemporalBiasCorrection:
    enabled: bool = True

    def fit(self, samples, minimum=30):
        """Return bias/scatter metadata; disabled is exactly zero correction."""
        if not self.enabled:
            return dict(bias=0.,scatter=0.,count=0,reliable=False,estimated_bias=0.,estimated_scatter=0.)
        return estimate_width_bias(samples,minimum)

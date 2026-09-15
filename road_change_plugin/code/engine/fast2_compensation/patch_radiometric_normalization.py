"""RGB -> grayscale, then background-only radiometric mapping after registration."""
from dataclasses import dataclass
import numpy as np


def q(values, fraction, default=np.nan):
    values=np.asarray(values);values=values[np.isfinite(values)]
    return float(np.quantile(values,fraction)) if values.size else default


@dataclass(frozen=True)
class PatchRadiometricNormalization:
    enabled: bool = True

    def grayscale(self, rgb):
        """Return float32 gray and valid mask; disabled never rescales intensities."""
        gray=np.mean(rgb,axis=0);valid=np.isfinite(gray)
        if not self.enabled:return np.nan_to_num(gray,nan=0.).astype(np.float32),valid
        low,high=q(gray,.1),q(gray,.9)
        if not np.isfinite(high-low) or high-low<=1e-6:return np.zeros(gray.shape,np.float32),valid&False
        return np.nan_to_num(np.clip((gray-low)/(high-low),-1,2),nan=0.).astype(np.float32),valid

    def apply(self, a, b, mask):
        """Aligned gray arrays + background mask -> mapped b, gain/offset audit."""
        info=dict(gain=1.,offset=0.)
        if not self.enabled:return b,info
        alo,ahi=q(a[mask],.2),q(a[mask],.8);blo,bhi=q(b[mask],.2),q(b[mask],.8)
        if np.isfinite(ahi-alo) and bhi-blo>1e-6:
            gain=float(np.clip((ahi-alo)/(bhi-blo),2/3,1.5))
            offset=float(np.clip(q(a[mask],.5)-gain*q(b[mask],.5),-.25,.25))
            b=(b*gain+offset).astype(np.float32);info.update(gain=gain,offset=offset)
        return b,info

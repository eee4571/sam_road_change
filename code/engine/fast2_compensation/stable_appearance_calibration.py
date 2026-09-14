"""Stable raw-image descriptors -> cross-period appearance reference distribution."""
from dataclasses import dataclass
import numpy as np


def quantile(values,q,default=np.nan):
    v=np.asarray(values,dtype=float);v=v[np.isfinite(v)]
    return float(np.quantile(v,q)) if len(v) else default


def estimate_appearance(c):
    calibration={'count':len(c),'minimum':12,'evidence':'raw_image_primary',
                      'encoder_features':'unavailable_skipped'}
    for name in ('score_delta','anomaly','ncc','ssim','hog_distance','left','right'):
        values=np.asarray([quantile(p[name],.5) if name in ('left','right') else p[name] for p in c])
        median=quantile(values,.5);mad=quantile(np.abs(values-median),.5)
        calibration.update({name+'_median':median,name+'_mad':mad,
            name+'_low':quantile(values,.05),name+'_high':quantile(values,.95),
            name+'_count':int(np.isfinite(values).sum())})
    for side in range(2):
        values=[p['scores'][side] for p in c]
        calibration[f'{side}_road_low']=quantile(values,.1)
    return calibration

@dataclass(frozen=True)
class StableAppearanceCalibration:
    enabled: bool = True

    def fit(self, descriptors):
        """Existing control descriptors -> same-schema reference, no image changes.

        Disabled uses the identity comparison (zero difference, similarity one),
        no learned shift, spread or quantile. Real validity/counts are retained;
        controls are never fabricated to bypass the verifier's quality checks.
        """
        if self.enabled:return estimate_appearance(descriptors)
        reference=dict(count=len(descriptors),minimum=12,evidence='raw_image_primary',
                       encoder_features='unavailable_skipped')
        for name in ('score_delta','anomaly','ncc','ssim','hog_distance','left','right'):
            neutral=1. if name in ('ncc','ssim') else 0.
            for suffix in ('median','low','high'):reference[name+'_'+suffix]=neutral
            reference[name+'_mad']=0.
            reference[name+'_count']=sum(bool(np.isfinite(p[name]).any()) for p in descriptors)
        reference['0_road_low']=reference['1_road_low']=0.
        return reference

"""Saved width quality and chain-scale representative widths (no measurement)."""
import numpy as np
from scipy.ndimage import median_filter


def reliable_width(row):
    grade = str(row.get('quality_grade', row.get('width_quality', ''))).upper()
    if not grade or grade == 'NAN':
        return None
    width = float(row.get('width_m', 0))
    ratio = float(row.get('valid_ratio', 1))
    scatter = float(row.get('width_std', 0))
    return grade in ('A', 'B') and ratio >= .15 and scatter <= .6 * width


def stable_width(stations, values, reliable):
    """One robust width per junction chain, split only by sustained evidence.

    Uniform sampling makes the representative independent of feature splits.
    Low-quality holes interpolate from reliable samples before segmentation;
    they cannot create a new narrow section or terminate the road surface.
    """
    s = np.asarray(stations, float)
    v = np.asarray(values, float)
    good = np.asarray(reliable, bool) & np.isfinite(v) & (v > 0)
    if not good.any():
        valid = np.isfinite(v) & (v > 0)
        if not valid.any():
            raise ValueError('Road surface has no positive representative width')
        result = np.full(len(v), np.median(v[valid]))
        return result, dict(method='unverified_representative', reliable_fraction=0., sections=1)
    filled = np.interp(s, s[good], v[good])
    representative = float(np.median(filled))
    step = max(float(np.median(np.diff(s))), .1)
    # A short wobble cannot become a width zone. Use physical length, not the
    # number or order of source features, as the persistence criterion.
    minimum = max(24., 3. * representative)
    filtered = median_filter(filled, size=max(3, int(8. / step) | 1), mode='nearest')
    threshold = max(1.5, .20 * representative)
    window=max(2,int(np.ceil(minimum/step)))
    candidates=[]
    for k in range(window,len(s)-window):
        left,right=filtered[k-window:k],filtered[k:k+window]
        lm,rm=float(np.median(left)),float(np.median(right))
        if abs(lm-rm)<threshold:continue
        # Both plateaux need actual A/B observations, not interpolated holes.
        if good[k-window:k].mean()<.6 or good[k:k+window].mean()<.6:continue
        if np.quantile(abs(left-lm),.8)>threshold/2 or np.quantile(abs(right-rm),.8)>threshold/2:continue
        loss=float(np.mean(abs(left-lm))+np.mean(abs(right-rm)))
        candidates.append((loss,k))
    boundaries=[]
    for _,k in sorted(candidates):
        if all(abs(s[k]-s[j])>=minimum for j in boundaries):boundaries.append(k)
    edges=[0,*sorted(boundaries),len(s)]
    # Collapse a candidate if the resulting full sections do not actually have
    # different representative widths. Local noise cannot create an extra zone.
    changed=True
    while changed:
        changed=False
        for i in range(1,len(edges)-1):
            if abs(np.median(filled[edges[i-1]:edges[i]])-np.median(filled[edges[i]:edges[i+1]]))<threshold:
                edges.pop(i);changed=True;break
    zones=list(zip(edges[:-1],edges[1:]))
    out = np.empty(len(v))
    levels = []
    for a, b in zones:
        level = float(np.median(filled[a:b]))
        out[a:b] = level
        levels.append(level)
    # Monotone cubic transitions, zero slope at both ends; no width overshoot.
    for i in range(len(zones) - 1):
        k = zones[i][1]
        span = min(max(6., representative), (s[k] - s[zones[i][0]]) / 3,
                   (s[zones[i + 1][1] - 1] - s[k]) / 3)
        if span <= 0:
            continue
        selected = abs(s - s[k]) <= span
        t = np.clip((s[selected] - s[k] + span) / (2 * span), 0, 1)
        out[selected] = levels[i] + (levels[i + 1] - levels[i]) * t * t * (3 - 2 * t)
    return out, dict(method='junction_chain_representative', sections=len(zones),
                     reliable_fraction=float(good.mean()), filled_samples=int((~good).sum()),
                     width_before_range=[float(v.min()), float(v.max())],
                     width_after_range=[float(out.min()), float(out.max())])

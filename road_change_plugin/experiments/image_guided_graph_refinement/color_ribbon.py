"""RGB transverse ribbon descriptors, independent of SAM and axis-derived surfaces."""
import numpy as np


def descriptor_scales(width):
    """Fixed apertures from cached width; these are not new width measurements."""
    return sorted({max(3., min(40., width*factor)) for factor in (.5, .75, 1.)})


def color_features(rgb, fractions, independent_flanks=False):
    """Color separation from BOTH flanks relative to internal color dispersion.

    Values are normalized reflectance-like display intensities, not semantic
    labels. A uniformly colored roof can still score highly; graph guards remain
    necessary. This descriptor detects chromatic edges invisible in grayscale.
    """
    valid = np.isfinite(rgb).all(axis=(-2, -1))
    values = np.nan_to_num(rgb)
    interior = values[..., (fractions >= .2) & (fractions <= .8), :]
    center = np.median(interior, axis=-2)
    left = np.median(values[..., fractions < -.05, :], axis=-2)
    right = np.median(values[..., fractions > 1.05, :], axis=-2)
    dl, dr = center-left, center-right
    contrast = np.minimum(np.linalg.norm(dl, axis=-1), np.linalg.norm(dr, axis=-1))/np.sqrt(3)
    spread = np.quantile(np.linalg.norm(interior-center[..., None, :], axis=-1)/np.sqrt(3), .8, axis=-1)
    # Flanks must not merely be a single building/field step running through the
    # proposed axis: both color differences have to point in compatible directions.
    same = np.ones(valid.shape, bool) if independent_flanks else np.sum(dl*dr, axis=-1) > 0
    supported = valid & same & (contrast >= .10) & (contrast >= 2*spread+.025)
    same_material = np.linalg.norm(values-center[..., None, :], axis=-1)/np.sqrt(3) < .10
    material_center = np.sum(same_material*fractions, axis=-1)/np.maximum(same_material.sum(axis=-1), 1)
    centrality = np.exp(-((material_center-.5)/.25)**2)
    score = np.clip(contrast/.25, 0, 1)*np.exp(-spread/.12)*same*valid*centrality
    return score, supported, contrast, spread


def sample_color(evidence, probes, fractions, longitudinal_window=1):
    image = evidence.image
    if not getattr(evidence, 'multichannel_ribbon', False) or not hasattr(image, 'rgb'):
        return None
    if not hasattr(image, '_ribbon_rgb_range'):
        # One fixed normalization per ROI, never one stretch per candidate.
        sample = image.rgb[::8, ::8].reshape(-1, 3)
        lo, hi = np.nanquantile(sample, [.02, .98], axis=0)
        image._ribbon_rgb_range = (lo, np.maximum(hi-lo, 1.))
    lo, span = image._ribbon_rgb_range
    flat = probes.reshape(-1, 2)
    rgb = np.stack([image.sample(image.rgb[..., k], flat) for k in range(3)], axis=-1)
    rgb = np.clip((rgb-lo)/span, 0, 1).reshape(*probes.shape[:-1], 3)
    if longitudinal_window > 1:
        from scipy.ndimage import median_filter
        rgb = median_filter(rgb, size=(longitudinal_window,)+(1,)*(rgb.ndim-1), mode='nearest')
    return color_features(rgb, fractions, independent_flanks=getattr(evidence, 'independent_flanks', False))

"""Conservative station planning. Unknown evidence always requests exact work."""
import numpy as np
from shapely.ops import substring


def presence_mask(count, spacing, intervals):
    mask = np.zeros(count, dtype=bool)
    # Keep the original cells, including either boundary cell and one context cell.
    for start, end in intervals:
        first = max(0, int(np.floor(start / spacing))-1)
        last = min(count, int(np.ceil(end / spacing))+1)
        mask[first:last] = True
    return mask


def width_mask(axis, source, target, stations, widths, tolerance, config, rank_equal):
    """Certify stable blocks, not merely similar width_m values.

    Stored widths do not bound the surface/probability fusion. Without identical
    measurement support a nominally stable block still takes the exact path.
    """
    result = np.ones(len(stations), dtype=bool)
    if not rank_equal or not source.has_width_values or not target.has_width_values:
        return result
    ids = target.tree.query(axis, predicate='dwithin', distance=tolerance+1e-6)
    own = source.tree.query(axis, predicate='dwithin', distance=tolerance+1e-6)
    if len(ids) != 1 or len(own) != 1:
        return result
    other = target.lines[int(ids[0])]
    if not axis.equals_exact(other, 0.):
        return result
    p, q = source.probability, target.probability
    # This certificate is intentionally restricted to an aligned metric grid.
    if (p.crs != source.crs or q.crs != target.crs or p.crs != q.crs
            or p.dataset.transform != q.dataset.transform or p.dataset.shape != q.dataset.shape
            or p.divisor != q.divisor or p.dataset.transform.b != 0 or p.dataset.transform.d != 0):
        return result
    for start in range(0, len(stations), 64):
        stop = min(start+64, len(stations))
        segment = substring(axis, max(0., stations[start]-4), min(axis.length, stations[stop-1]+4))
        width_ids = target.width_tree.query(segment, predicate='dwithin', distance=3)
        values = target.width_values[width_ids]
        source_ids = source.width_tree.query(segment, predicate='dwithin', distance=3)
        source_values = source.width_values[source_ids]
        if (not len(values) or not np.isfinite(values).all() or (values <= 0).any()
                or not len(source_values) or not np.isfinite(source_values).all() or (source_values <= 0).any()
                or not np.isfinite(widths[start:stop]).all()
                or np.max(np.abs(values[:, None]-widths[None, start:stop])) >= config.absolute_change):
            continue
        radius = config.normal_half_length+5
        left, bottom, right, top = segment.bounds
        if not source.surface(segment, radius).equals(target.surface(segment, radius)):
            continue
        # Compare raster evidence once per block, without station cross sections.
        cols, rows = p.inverse * (np.array([left-radius, right+radius]),
                                  np.array([bottom-radius, top+radius]))
        c0, c1 = max(0, int(np.floor(cols.min()))), min(p.dataset.width, int(np.ceil(cols.max()))+1)
        r0, r1 = max(0, int(np.floor(rows.min()))), min(p.dataset.height, int(np.ceil(rows.max()))+1)
        if c1 <= c0 or r1 <= r0 or (c1-c0)*(r1-r0) > 4_000_000:
            continue
        import rasterio
        window = rasterio.windows.Window(c0, r0, c1-c0, r1-r0)
        a = p._ram[r0:r1, c0:c1] if p._ram is not None else p.dataset.read(1, window=window, masked=True)
        b = q._ram[r0:r1, c0:c1] if q._ram is not None else q.dataset.read(1, window=window, masked=True)
        if np.ma.allequal(a, b) and np.array_equal(np.ma.getmaskarray(a), np.ma.getmaskarray(b)):
            result[start:stop] = False
    # Exact neighbours preserve run boundaries and the existing one-sample gap rule.
    candidates = np.flatnonzero(result)
    for offset in (-2, -1, 1, 2):
        result[np.clip(candidates+offset, 0, len(result)-1)] = True
    return result


class BatchSections:
    """Feed pre-sampled profiles to the unchanged paired-width estimator."""
    def __init__(self, sampler, centers, normals, radius):
        self.sampler = sampler
        self.profiles = {}
        distances = np.arange(-radius, radius+.25, .5)
        for start in range(0, len(centers), 256):
            points = centers[start:start+256]
            directions = normals[start:start+256]
            xy = np.array([(p.x, p.y) for p in points])
            coords = xy[:, None, :]+np.asarray(directions)[:, None, :]*distances[None, :, None]
            values = sampler._values_at(coords[..., 0].ravel(), coords[..., 1].ravel()).reshape(len(points), -1)
            for point, normal, row in zip(points, directions, values):
                self.profiles[self.key(point, normal, radius)] = dict(distance=distances, probability=row,
                    valid_mask=np.isfinite(row), sample_step=.5)

    @staticmethod
    def key(point, normal, radius):
        return (point.x, point.y, float(normal[0]), float(normal[1]), radius)

    def percentile_rank(self, value):
        return self.sampler.percentile_rank(value)

    def sample_cross_section(self, center, normal, geometry_crs, *, search_radius):
        profile = self.profiles.get(self.key(center, normal, search_radius))
        if profile is not None:
            return profile
        return self.sampler.sample_cross_section(center, normal, geometry_crs, search_radius=search_radius)

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


def width_mask(axis, source, target, stations, widths, tolerance, config):
    """Skip only well matched blocks with a substantial width stability margin.

    Stored widths are a prefilter, not a replacement width estimator. Missing
    coverage, competing tracks, inconsistent surfaces and threshold neighbours
    retain the original exact stations.
    """
    from shapely import union_all
    result = np.ones(len(stations), dtype=bool)
    if not source.has_width_values or not target.has_width_values or not axis.is_simple:
        return result
    own = source.tree.query(axis, predicate='dwithin', distance=1e-6)
    own = [int(i) for i in own if source.lines[int(i)].equals(axis)]
    if len(own) != 1:
        return result
    pixel = max(source.probability.pixel_size, target.probability.pixel_size)
    for start in range(0, len(stations), 16):
        stop = min(start+16, len(stations))
        segment = substring(axis, max(0., stations[start]-4), min(axis.length, stations[stop-1]+4))
        ids = target.tree.query(segment, predicate='dwithin', distance=tolerance+1e-6)
        # Nearby competitors and junction branches always use precise matching.
        if len(ids) != 1 or len(source.tree.query(segment, predicate='dwithin', distance=tolerance+1e-6)) != 1:
            continue
        other = target.lines[int(ids[0])]
        if not other.is_simple or source.junction.intersects(segment) or target.junction.intersects(segment):
            continue
        matches = [target.match(axis, float(stations[i]), tolerance, float(widths[i]))
                   for i in sorted({start, (start+stop-1)//2, stop-1})]
        if any(m is None or not m['reliable'] or m['target'] != int(ids[0])
               or m['direction'] < .98 or m['coverage'] < .95
               or m['distance'] > min(tolerance*.5, 1.) for m in matches):
            continue
        if np.ptp([m['distance'] for m in matches]) > .5:
            continue
        reverse = [source.match(other, m['station'], tolerance, target.width(other.interpolate(m['station']))) for m in matches]
        if any(m is None or not m['reliable'] or m['target'] != own[0] for m in reverse):
            continue
        ends = [other.project(axis.interpolate(s)) for s in
                (max(0., stations[start]-4), min(axis.length, stations[stop-1]+4))]
        paired = substring(other, min(ends), max(ends))
        if paired.length < .95*segment.length or paired.length > 1.05*segment.length:
            continue
        profiles = []
        for scene, part in ((source, segment), (target, paired)):
            indexes = scene.width_tree.query(part, predicate='dwithin', distance=3)
            values = np.asarray(scene.width_values[indexes], dtype=float)
            if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
                break
            support = union_all(scene.width_geometries[indexes]).buffer(.75)
            if part.intersection(support).length < .99*part.length:
                break
            profiles.append(values)
        if len(profiles) != 2:
            continue
        low = min(float(v.min()) for v in profiles)
        high = max(float(v.max()) for v in profiles)
        # Stay below BOTH formal thresholds with at least a 50% reserve,
        # additionally accounting for resolution and within-block variation.
        threshold = min(config.absolute_change, config.relative_change*low)
        budget = .5*threshold-max(.25, .25*pixel)
        if budget <= 0 or high-low >= budget:
            continue
        consistent = True
        for scene, part, values in zip((source, target), (segment, paired), profiles):
            width = float(np.median(values))
            corridor = part.buffer(width/2, cap_style='flat')
            window = part.buffer(width/2+config.absolute_change*2, cap_style='flat')
            surface = scene.surface(part, width+config.absolute_change*2).intersection(window)
            # Area per longitudinal metre detects stale nominal widths without
            # demanding equal polygons or reading/comparing raster windows.
            if (not scene.valid.covers(corridor) or
                    surface.symmetric_difference(corridor).area/max(part.length, 1e-9) >= budget):
                consistent = False
                break
        if consistent:
            result[start:stop] = False
    # Retain exact neighbours for the unchanged run/gap rules.
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

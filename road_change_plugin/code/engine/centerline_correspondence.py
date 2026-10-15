"""Source-independent, reciprocal matching of road axes in image pixels.

The 64 px association window and 45 degree direction gate establish candidate
correspondence, not positional acceptance. The separate 4 px acceptance limit
must never censor distances in the mean. Unmatched lengths remain explicit.
"""
import numpy as np
from scipy.spatial import cKDTree


METHOD = 'reciprocal_road_segments_v1'
SEARCH_PX = 64.0
SAMPLE_PX = 1.0
DIRECTION_COS = np.cos(np.pi / 4)


def axis_samples(mask, origin=(0., 0.)):
    """Trace every skeleton edge, including branches and cycles, and sample it.

    Samples carry arc-length weights; diagonal pixels do not count as horizontal
    pixels. Paths terminate at junctions so tangents do not cross into a branch.
    """
    pixels = set(map(tuple, np.argwhere(mask)))
    neighbors = {}
    for y, x in sorted(pixels):
        adjacent = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                other = (y + dy, x + dx)
                if not (dy or dx) or other not in pixels:
                    continue
                # Do not create triangles across an existing orthogonal edge.
                if dy and dx and ((y + dy, x) in pixels or (y, x + dx) in pixels):
                    continue
                adjacent.append(other)
        neighbors[(y, x)] = adjacent
    visited = set()
    paths = []

    def trace(start, following):
        path = [start]
        previous, current = start, following
        while True:
            edge = tuple(sorted((previous, current)))
            if edge in visited:
                break
            visited.add(edge)
            path.append(current)
            if len(neighbors[current]) != 2:
                break
            nxt = next(p for p in neighbors[current] if p != previous)
            previous, current = current, nxt
        paths.append(path)

    for start in neighbors:
        if len(neighbors[start]) != 2:
            for following in neighbors[start]:
                trace(start, following)
    for start in neighbors:  # Remaining edges are closed cycles.
        for following in neighbors[start]:
            if tuple(sorted((start, following))) not in visited:
                trace(start, following)

    batches = []
    for path in paths:
        if len(path) < 2:
            continue
        xy = np.asarray(path, dtype=float)[:, ::-1] + np.asarray(origin)
        station = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        length = station[-1]
        count = max(1, int(np.ceil(length / SAMPLE_PX)))
        weight = length / count
        positions = (np.arange(count) + .5) * weight

        def interpolate(s):
            return np.column_stack([np.interp(s, station, xy[:, i]) for i in (0, 1)])

        points = interpolate(positions)
        tangent = interpolate(np.minimum(length, positions + 3.)) - interpolate(np.maximum(0., positions - 3.))
        norm = np.linalg.norm(tangent, axis=1)
        valid = norm > 1e-9
        batches.append(np.column_stack((points[valid], tangent[valid] / norm[valid, None],
                                        np.full(valid.sum(), weight))))
    return np.concatenate(batches) if batches else np.empty((0, 5))


def corresponding_distances(predicted, truth):
    """Return distances and length weights on mutually supported road segments.

    Reciprocal nearest locations reject extensions projecting to one endpoint
    and branches for which the reference already corresponds to the main road.
    A 1.5-sample reciprocal allowance handles discrete sampling, independent of
    the measured lateral displacement. Ambiguous equal-distance roads are not
    silently chosen as the correct road.
    """
    if not len(predicted) or not len(truth):
        return np.empty(0), np.empty(0)

    def nearest(source, target):
        tree = cKDTree(target[:, :2])
        indices = np.full(len(source), -1, dtype=int)
        for i, sample in enumerate(source):
            candidates = np.asarray(tree.query_ball_point(sample[:2], SEARCH_PX), dtype=int)
            if not len(candidates):
                continue
            candidates = candidates[np.abs(target[candidates, 2:4] @ sample[2:4]) >= DIRECTION_COS]
            if not len(candidates):
                continue
            distance = np.linalg.norm(target[candidates, :2] - sample[:2], axis=1)
            closest = distance.min()
            tied = candidates[np.isclose(distance, closest, atol=1e-8, rtol=0)]
            # Adjacent samples can tie on one road; spatially distinct roads
            # must remain unmatched rather than depending on feature order.
            if np.max(np.linalg.norm(target[tied, :2] - target[tied[0], :2], axis=1)) > 2 * SAMPLE_PX:
                continue
            indices[i] = tied[np.lexsort((target[tied, 1], target[tied, 0]))[0]]
        return indices

    forward, backward = nearest(predicted, truth), nearest(truth, predicted)
    chosen = np.flatnonzero(forward >= 0)
    chosen = chosen[backward[forward[chosen]] >= 0]
    reciprocal = backward[forward[chosen]]
    chosen = chosen[np.linalg.norm(predicted[chosen, :2] - predicted[reciprocal, :2], axis=1) <= 1.5 * SAMPLE_PX]
    return np.linalg.norm(predicted[chosen, :2] - truth[forward[chosen], :2], axis=1), predicted[chosen, 4]

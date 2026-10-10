"""Orientation-state RGB ribbon routing for interrupted streets with real turns."""
import heapq
import numpy as np
from shapely.geometry import LineString
from rgb_route import ribbon_cost_features
from color_ribbon import sample_color


def turning_route(start, end, start_heading, end_heading, evidence, pavement_rgb=None):
    start, end = np.asarray(start, float), np.asarray(end, float)
    chord = np.linalg.norm(end-start)
    resolution = 2.
    lo = np.minimum(start, end)-30
    hi = np.maximum(start, end)+30
    cols, rows = np.ceil((hi-lo)/resolution).astype(int)+1
    xx, yy = np.meshgrid(lo[0]+np.arange(cols)*resolution, lo[1]+np.arange(rows)*resolution)
    xy = np.c_[xx.ravel(), yy.ravel()]
    steps = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])
    headings = steps/np.linalg.norm(steps, axis=1)[:, None]
    costs = np.ones((rows*cols, 8))*12
    supports = np.zeros_like(costs)
    fractions = np.linspace(-.5, 1.5, 25)
    rgb = np.column_stack([evidence.image.sample(evidence.image.rgb[..., k], xy) for k in range(3)])/255.
    vegetation = np.maximum(0., 2*rgb[:, 1]-rgb[:, 0]-rgb[:, 2]-.05)
    appearance_cost = np.zeros(len(xy))
    if pavement_rgb is not None and np.isfinite(pavement_rgb).all():
        brightness = pavement_rgb.mean(axis=1)
        low, high = np.quantile(brightness, [.1, .9])
        intensity = rgb.mean(axis=1)
        # Bright roofs and differently colored yards are not continuations of
        # the observed entrance pavement, even when they have sharp edges.
        chroma = rgb/np.maximum(rgb.sum(axis=1, keepdims=True), .05)
        reference = pavement_rgb/np.maximum(pavement_rgb.sum(axis=1, keepdims=True), .05)
        appearance_cost = 40*(np.maximum(low-.08-intensity, 0)+np.maximum(intensity-high-.08, 0))
        appearance_cost += 30*np.linalg.norm(chroma-np.median(reference, axis=0), axis=1)
    for direction in range(4):
        normal = np.array([-headings[direction, 1], headings[direction, 0]])
        best = np.zeros(len(xy))
        for aperture in (4., 6., 9., 13.):
            probes = xy[:, None]+normal*(fractions[None, :, None]-.5)*aperture
            gray = evidence.image.sample(evidence.image.gray, probes.reshape(-1, 2)).reshape(len(xy), -1)
            score, valid, _, _ = ribbon_cost_features(gray, fractions)
            color = sample_color(evidence, probes, fractions)
            if color is not None:
                score = np.maximum(score, color[0])
            best = np.maximum(best, np.where(valid, score, 0))
        # Low SAM has no cost. RGB and bilateral structure supply all negatives.
        for d in (direction, direction+4):
            costs[:, d] = 1+9*(1-best)+20*np.nan_to_num(vegetation, nan=1)+np.nan_to_num(appearance_cost, nan=50)
            supports[:, d] = best
    costs = costs.reshape(rows, cols, 8)
    supports = supports.reshape(rows, cols, 8)
    a = np.rint((start-lo)/resolution).astype(int)
    b = np.rint((end-lo)/resolution).astype(int)
    initial_direction = int(np.argmax(headings@start_heading))
    arrival_direction = int(np.argmax(headings@end_heading))
    initial = (int(a[1]), int(a[0]), initial_direction)
    best = {initial: 0.}; previous = {}
    queue = [(chord, 0., initial)]
    final = None
    while queue:
        _, value, state = heapq.heappop(queue)
        if value > best[state]+1e-8:
            continue
        r, c, heading = state
        if r == b[1] and c == b[0] and min((heading-arrival_direction)%8, (arrival_direction-heading)%8) <= 1:
            final = state; break
        for turn in (-1, 0, 1):
            d = (heading+turn)%8
            dc, dr = steps[d]; nr, nc = r+dr, c+dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            step_length = resolution*np.linalg.norm(steps[d])
            next_value = value+step_length*costs[nr, nc, d]+abs(turn)*3
            next_state = (nr, nc, d)
            if next_value >= best.get(next_state, np.inf):
                continue
            best[next_state] = next_value; previous[next_state] = state
            heuristic = resolution*np.hypot(nr-b[1], nc-b[0])
            heapq.heappush(queue, (next_value+heuristic, next_value, next_state))
    if final is None:
        return None, dict(reason='no_oriented_route')
    path = [final]
    while path[-1] != initial:
        path.append(previous[path[-1]])
    path.reverse()
    coords = np.array([[lo[0]+c*resolution, lo[1]+r*resolution] for r, c, _ in path])
    coords[0], coords[-1] = start, end
    line = LineString(coords).simplify(1.5)
    strength = np.array([supports[r, c, d] for r, c, d in path])
    return line, dict(route_strength_mean=float(strength.mean()), route_length_m=line.length,
                      chord_length_m=float(chord), orientation_constrained_during_search=True)

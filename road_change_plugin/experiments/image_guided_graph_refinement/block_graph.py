"""Block-scale vector graph fitting, with shared junction variables.

The optimization unit is a connected street block, not one defective edge.
RGB supplies transverse observations; long-range direction and curvature terms
regularize the entire graph. Original geometry is a prior, not a pinned answer.
"""
import numpy as np
import geopandas as gpd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from shapely import union_all, points, distance
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, substring
from shapely.strtree import STRtree
import pandas as pd

from graph import build
from evidence import stations
from rgb_route import ribbon_cost_features
from color_ribbon import sample_color, descriptor_scales
from local_graph_repair import seed_chains
from medial_repair import line_parts, pin_axis_contacts
from graph_transaction import validate
from engine.canonical_road_surface import _adjacency


def appearance_scales(width):
    """Fixed image descriptor scales include narrow roads missed by cached widths."""
    return sorted(set([3., 5., 8.]+list(descriptor_scales(width))))


def complete_streets(frame, evidence, roi):
    """Fill interruptions using a whole approach, rather than a noisy end segment.

    A local pavement appearance model is built from the two observed approaches.
    Chromaticity permits illumination changes, while bilateral RGB structure and
    limited holes still have to support the actual intervening corridor.
    """
    from evidence import longest_hole
    from graph import Candidate
    from refine import safe_gap
    edges = seed_chains(build(frame))
    nodes = _adjacency(edges, set(range(len(edges))))
    tips = []
    for node, ends in nodes.items():
        if len(ends) != 1 or not roi.buffer(-5).covers(Point(node[1:])):
            continue
        i, end = ends[0]; edge = edges[i]
        if edge.axis.length < 25:
            continue
        distances = np.linspace(3, min(40, edge.axis.length), 12)
        if end:
            distances = edge.axis.length-distances
        xy = np.array([edge.axis.interpolate(s).coords[0] for s in distances])
        _, _, vt = np.linalg.svd(xy-xy.mean(axis=0), full_matrices=False)
        v = vt[0]
        if v@(np.asarray(node[1:])-xy.mean(axis=0)) < 0:
            v = -v
        normal = np.array([-v[1], v[0]])
        if np.std(xy@normal) > 2.5:
            continue
        tips.append((node, i, v, xy))
    audit = []; used = set(); result = frame
    image = evidence.image
    def rgb(xy):
        return np.column_stack([image.sample(image.rgb[..., k], xy) for k in range(3)])/255.
    pairs = []
    for i, (a, ai, av, ax) in enumerate(tips):
        for j in range(i):
            b, bi, bv, bx = tips[j]
            if edges[ai].level != edges[bi].level or ai == bi:
                continue
            delta = np.asarray(b[1:])-a[1:]; length = np.linalg.norm(delta)
            if not 2 <= length <= 120:
                continue
            v = delta/length
            if min(av@v, bv@-v) < -.1:
                continue
            pairs.append((length, i, j))
    for length, i, j in sorted(pairs):
        a, ai, av, ax = tips[i]; b, bi, bv, bx = tips[j]
        if a in used or b in used:
            continue
        line = LineString([a[1:], b[1:]])
        if not roi.buffer(-2).covers(line):
            continue
        from street_route import turning_route
        # Each approach is a separate material hypothesis. Pooling a shaded
        # approach with a bright one makes the permitted intensity range include
        # roofs in between; it was the source of a control-ROI shortcut.
        route_hypotheses = []
        for side, approach in enumerate((ax, bx)):
            reference = rgb(approach)
            trial, info = turning_route(a[1:], b[1:], av, -bv, evidence, reference)
            if trial is None or trial.length > length*1.8+10 or not roi.buffer(-2).covers(trial):
                continue
            trial_rgb = rgb(stations(trial, 2)[1])
            low, high = np.nanquantile(reference.mean(axis=1), [.1, .9])
            material_fraction = float(np.mean((trial_rgb.mean(axis=1) >= low-.08)
                                             & (trial_rgb.mean(axis=1) <= high+.08)))
            info.update(appearance_hypothesis=side, independent_material_fraction=material_fraction)
            route_hypotheses.append((material_fraction, info['route_strength_mean']-.001*trial.length,
                                     trial, info, reference))
        if route_hypotheses:
            _, _, route, route_info, appearance = max(route_hypotheses, key=lambda h: h[:2])
        else:
            route, route_info, appearance = None, dict(reason='no_single_material_oriented_route'), rgb(ax)
        if route is None or route.length > length*1.8+10 or not roi.buffer(-2).covers(route):
            audit.append(dict(action='complete_street', applied=False, candidate_wkt=line.wkt,
                              evidence=route_info, reason='no_supported_bounded_turning_corridor'))
            continue
        line = route
        _, xy, _, normals = stations(line, 2)
        chroma = appearance/np.maximum(appearance.sum(axis=1, keepdims=True), .05)
        center = rgb(xy)
        target = center/np.maximum(center.sum(axis=1, keepdims=True), .05)
        similarity = np.linalg.norm(target-np.nanmedian(chroma, axis=0), axis=1) < .09
        green = 2*center[:, 1]-center[:, 0]-center[:, 2] > .1
        observations = []
        width = float(np.median(np.r_[edges[ai].widths, edges[bi].widths]))
        for aperture in appearance_scales(width):
            left, right = rgb(xy+normals*aperture*.55), rgb(xy-normals*aperture*.55)
            dl, dr = center-left, center-right
            bilateral = np.minimum(np.linalg.norm(dl, axis=1), np.linalg.norm(dr, axis=1)) > .10
            supported = similarity & ~green & bilateral & np.isfinite(center).all(axis=1)
            observations.append((float(supported.mean()), longest_hole(supported, line.length), aperture))
        support, hole, aperture = max(observations)
        current = build(result)
        # Crossings are assessed against the actual current graph, not stale ids.
        endpoint_disks = Point(a[1:]).buffer(.05).union(Point(b[1:]).buffer(.05))
        crossing = any(not line.intersection(e.axis).difference(endpoint_disks).is_empty for e in current)
        low, high = np.nanquantile(appearance.mean(axis=1), [.1, .9])
        brightness = center.mean(axis=1)
        same_pavement = similarity & ~green & (brightness >= low-.08) & (brightness <= high+.08)
        dispersion = np.quantile(np.linalg.norm(center-np.median(center, axis=0), axis=1)/np.sqrt(3), .8)
        # Two existing access streets can meet on a paved forecourt where there
        # are no bilateral curbs. This is a distinct junction hypothesis, with
        # continuous matching material required rather than a weaker ribbon gate.
        forecourt = (line.length <= 75 and same_pavement.mean() >= .95 and dispersion <= .08
                     and longest_hole(same_pavement, line.length) <= 4)
        material_hole = longest_hole(same_pavement, line.length)
        material_passed = same_pavement.mean() >= .85 and material_hole <= min(8., line.length*.2)
        passed = not crossing and material_passed and ((support >= .65 and hole <= min(18, line.length*.3)) or forecourt)
        record = dict(action='complete_street', applied=False, new_wkt=line.wkt,
                      evidence=dict(route_info, pavement_chromaticity_fraction=float(similarity.mean()),
                                    bilateral_continuity=support, maximum_hole_m=hole, descriptor_aperture_m=aperture,
                                    crossing_detected=crossing, gap_length_m=length),
                      reason='long_approach_direction_and_same_pavement_bilateral_continuation')
        record['evidence'].update(forecourt_hypothesis=bool(forecourt), same_pavement_fraction=float(same_pavement.mean()),
                                  pavement_color_dispersion=float(dispersion), maximum_material_hole_m=material_hole,
                                  material_continuity_passed=bool(material_passed))
        if forecourt:
            record['reason'] = 'existing_access_streets_join_on_continuous_matching_forecourt_pavement'
        if passed:
            template = frame.iloc[edges[ai].sources[0]].to_dict()
            addition = gpd.GeoDataFrame([dict(template, geometry=line, repair_role='complete_street')], crs=frame.crs)
            candidate = gpd.GeoDataFrame(pd.concat([result, addition], ignore_index=True), crs=frame.crs)
            guard = validate(result, candidate, line.buffer(width).intersection(roi), evidence)
            record['transaction_guard'] = guard
            if guard['passed']:
                result = candidate; record['applied'] = True; used.update([a, b])
        audit.append(record)
    return result, audit


def pavement_inside(face, evidence):
    """A real ring encloses a different material; an alias encloses road itself."""
    boundary = LineString(face.exterior.coords)
    edge_xy = stations(boundary, 1.5)[1]
    x0, y0, x1, y1 = face.bounds
    xx, yy = np.meshgrid(np.arange(x0, x1+.8, .8), np.arange(y0, y1+.8, .8))
    xy = np.c_[xx.ravel(), yy.ravel()]
    xy = xy[face.buffer(-.25).covers(points(xy))]
    if len(xy) < 4:
        return dict(consistent=False, reason='insufficient_interior_pixels')
    image = evidence.image
    a = np.column_stack([image.sample(image.rgb[..., k], edge_xy) for k in range(3)])/255.
    b = np.column_stack([image.sample(image.rgb[..., k], xy) for k in range(3)])/255.
    if np.isfinite(a).mean() < .98 or np.isfinite(b).mean() < .98:
        return dict(consistent=False, reason='invalid_rgb')
    median = np.median(a, axis=0)
    discrepancy = np.linalg.norm(b-median, axis=1)/np.sqrt(3)
    spread = np.quantile(np.linalg.norm(a-median, axis=1)/np.sqrt(3), .8)
    agree = float(np.mean(discrepancy < min(.16, max(.07, spread))))
    vegetation = float(np.mean(2*b[:, 1]-b[:, 0]-b[:, 2] > .1))
    return dict(consistent=agree >= .85 and vegetation < .15,
                interior_same_material_fraction=agree, interior_vegetation_fraction=vegetation,
                boundary_color_spread=float(spread), interior_samples=len(xy))


def corridor_spine(face, anchors):
    """A through-axis for an elongated same-pavement face, with every entry fixed."""
    rectangle = np.asarray(face.minimum_rotated_rectangle.exterior.coords)[:4]
    sides = np.roll(rectangle, -1, axis=0)-rectangle
    lengths = np.linalg.norm(sides, axis=1)
    if lengths.max() < 3*max(lengths.min(), 1):
        return None
    tangent = sides[lengths.argmax()]/lengths.max()
    normal = np.array([-tangent[1], tangent[0]])
    origin = np.asarray(face.centroid.coords[0])
    coords = np.array([n[1:] for n in anchors])
    positions = (coords-origin)@tangent
    if np.ptp(positions) < .6*lengths.max():
        return None
    first, last = coords[positions.argmin()], coords[positions.argmax()]
    samples = np.linspace(positions.min(), positions.max(), max(4, int(np.ptp(positions)/2)))
    middle = []
    for s in samples:
        center = origin+s*tangent
        cut = LineString([center-normal*lengths.max(), center+normal*lengths.max()]).intersection(face)
        parts = line_parts(cut)
        if not parts:
            middle.append(center)
        else:
            middle.append(np.asarray(max(parts, key=lambda p: p.length).interpolate(.5, normalized=True).coords[0]))
    xy = np.asarray(middle)
    # Fixed cap contacts join gradually, not through a perpendicular connector.
    taper_a = np.clip(1-(samples-samples[0])/12, 0, 1)
    taper_b = np.clip(1-(samples[-1]-samples)/12, 0, 1)
    xy += taper_a[:, None]*(first-xy[0])+taper_b[:, None]*(last-xy[-1])
    xy[0], xy[-1] = first, last
    axis = LineString(xy).simplify(.25)
    branches, contacts = [], []
    for coord in coords:
        point = Point(coord); target = axis.interpolate(axis.project(point))
        if point.distance(target) > .01:
            branches.append(LineString([point, target])); contacts.append(target.coords[0])
    axis = pin_axis_contacts(axis, contacts)
    return [axis]+branches


def consolidate_junctions(frame, evidence, roi, corridor_mode=False):
    """Jointly replace aliases inside one paved patch while retaining all arms."""
    result = frame.copy()
    audit = []
    for iteration in range(30):
        edges = build(result)
        nodes = _adjacency(edges, set(range(len(edges))))
        faces = [p for level in {e.level for e in edges}
                 for p in polygonize(union_all([e.axis for e in edges if e.level == level]))
                 if roi.buffer(-5).covers(p) and p.length <= (1000 if corridor_mode else 180)]
        changed = False
        for face in sorted(faces, key=lambda p: p.area):
            ids = {i for i, e in enumerate(edges) if face.buffer(.01).covers(e.axis)}
            if not ids:
                continue
            width = float(np.median([np.median(edges[i].widths) for i in ids]))
            # This is only proposal scale. Positive interior evidence is mandatory.
            if 2*face.area/max(face.length, 1) > (12. if corridor_mode else min(8., width*.45)):
                continue
            evidence_row = pavement_inside(face, evidence)
            if not evidence_row['consistent']:
                continue
            anchors = sorted({n for i in ids for n in edges[i].nodes
                              if any(k not in ids for k, _ in nodes[n])}, key=str)
            if len(anchors) < 2 or len({edges[i].level for i in ids}) != 1:
                continue
            canonical = corridor_spine(face, anchors) if corridor_mode else None
            if canonical is not None:
                new = canonical
            elif face.length > 180:
                continue
            elif len(anchors) == 2:
                ring = LineString(face.exterior.coords)
                loc = sorted(ring.project(Point(n[1:])) for n in anchors)
                a = substring(ring, *loc)
                b = LineString((list(substring(ring, loc[1], ring.length).coords)+
                                list(substring(ring, 0, loc[0]).coords)[1:])[::-1])
                t = np.linspace(0, 1, max(4, int(max(a.length, b.length)/3)))
                xy = (np.array([a.interpolate(s, normalized=True).coords[0] for s in t])+
                      np.array([b.interpolate(s, normalized=True).coords[0] for s in t]))/2
                new = [LineString(xy).simplify(1.)]
            else:
                # Least-squares intersection of incoming long-arm tangents.
                normals, offsets = [], []
                for n in anchors:
                    for k, end in nodes[n]:
                        if k in ids:
                            continue
                        line = edges[k].axis
                        start = np.asarray(n[1:])
                        near = np.asarray(line.interpolate(min(15, line.length) if end == 0 else max(0, line.length-15)).coords[0])
                        v = near-start; v /= max(np.linalg.norm(v), 1e-9)
                        normal = np.array([-v[1], v[0]])
                        normals.append(normal); offsets.append(normal@start)
                if len(normals) < 2:
                    continue
                center = np.linalg.lstsq(normals, offsets, rcond=None)[0]
                if not face.buffer(2).covers(Point(center)):
                    center = np.asarray(face.representative_point().coords[0])
                new = [LineString([n[1:], center]) for n in anchors if Point(n[1:]).distance(Point(center)) > .01]
            rows = []
            for i, e in enumerate(edges):
                if i not in ids:
                    rows.append(dict(result.iloc[e.sources[0]], geometry=e.axis))
            template = result.iloc[edges[min(ids)].sources[0]].to_dict()
            rows.extend(dict(template, geometry=g, repair_role='block_junction') for g in new)
            candidate = gpd.GeoDataFrame(rows, geometry='geometry', crs=frame.crs)
            window = face.buffer(max(3, width*.5)).intersection(roi)
            guard = validate(result, candidate, window, evidence)
            audit.append(dict(action='canonical_corridor' if canonical is not None else 'block_junction_rebuild', applied=guard['passed'],
                              old_wkt=union_all([edges[i].axis for i in ids]).wkt,
                              new_wkt=union_all(new).wkt, evidence=evidence_row,
                              transaction_guard=guard, external_contacts=[list(n[1:]) for n in anchors],
                              reason='one_continuous_pavement_ribbon_single_axis_all_external_entries_retained' if canonical is not None else
                              'single_pavement_patch_joint_graph_rebuild_not_perimeter_deletion'))
            if guard['passed']:
                result = candidate
                changed = True
                break  # regenerate domains after a graph change
        if not changed:
            break
    return result, audit


def rgb_observations(line, width, evidence):
    """Transverse RGB observations at fixed cached apertures, not width estimates."""
    _, xy, tangent, normal = stations(line, 8.)
    offsets = np.arange(-6., 6.1, 1.)
    fractions = np.linspace(-.5, 1.5, 31)
    best = np.zeros((len(xy), len(offsets)))
    for aperture in appearance_scales(width):
        probes = xy[:, None, None]+normal[:, None, None]*(offsets[None, :, None, None]+
                    (fractions[None, None, :, None]-.5)*aperture)
        gray = evidence.image.sample(evidence.image.gray, probes.reshape(-1, 2)).reshape(len(xy), len(offsets), -1)
        scores, valid, _, _ = ribbon_cost_features(gray, fractions)
        color = sample_color(evidence, probes, fractions)
        if color is not None:
            scores = np.maximum(scores, color[0])
        best = np.maximum(best, np.where(valid, scores, 0))
    # Color variation and vegetation penalties prevent fitting a tree row as a
    # road solely because it forms a ribbon. Darkness is not a road reward.
    centers = xy[:, None]+normal[:, None]*offsets[None, :, None]
    rgb = np.stack([evidence.image.sample(evidence.image.rgb[..., k], centers.reshape(-1, 2)).reshape(len(xy), -1)
                    for k in range(3)], axis=-1)/255.
    green = np.maximum(0, 2*rgb[..., 1]-rgb[..., 0]-rgb[..., 2])
    score = np.nan_to_num(best)-2*green-.12*(offsets/6)**2
    selected = score.argmax(axis=1)
    shift = offsets[selected]
    confidence = np.maximum(0., score[np.arange(len(xy)), selected]-.15)
    target = xy+normal*shift[:, None]
    return xy, tangent, normal, target, confidence


def joint_fit(frame, evidence, roi, locked=None):
    """One sparse solve shares all junctions and long-range direction constraints."""
    edges = build(frame)
    locked = set() if locked is None else set(locked)
    nodes = _adjacency(edges, set(range(len(edges))))
    coords, lookup, paths, observations = [], {}, [], []
    vertex_edges, vertex_tangents = {}, {}
    pinned = set()
    for e in edges:
        xy, tangent, normals, target, confidence = rgb_observations(e.axis, float(np.median(e.widths)), evidence)
        ids = []
        for j, p in enumerate(xy):
            key = (e.level, *np.round(p, 6))
            if j not in (0, len(xy)-1):
                key = (len(paths), j)
            if key not in lookup:
                lookup[key] = len(coords); coords.append(p)
            k = lookup[key]; ids.append(k)
            vertex_edges.setdefault(k, set()).add(len(paths))
            vertex_tangents[k] = tangent[j]
            if len(paths) in locked or e.axis.length < 16 or e.axis.is_ring or not roi.buffer(-3).covers(e.axis) or (j in (0, len(xy)-1) and len(nodes[e.nodes[int(j > 0)]]) == 1):
                pinned.add(k)
            observations.append((k, normals[j], target[j], confidence[j]))
        paths.append(ids)
    original = np.asarray(coords)
    origin = original.mean(axis=0)
    # Coordinates are centered for numerical conditioning of sparse least squares.
    rr, cc, vv, values = [], [], [], []
    def add(terms, value, weight):
        row = len(values); values.append(value*weight)
        for col, coefficient in terms:
            rr.append(row); cc.append(col); vv.append(coefficient*weight)
    for k, p in enumerate(original-origin):
        for d in (0, 1):
            add([(2*k+d, 1)], p[d], 1000 if k in pinned else .35)
    for k, normal, target, confidence in observations:
        if k not in pinned:
            add([(2*k, normal[0]), (2*k+1, normal[1])], (target-origin)@normal, confidence*1.2)
    # A shared image observation must not pull two distinct nearby streets onto
    # each other. Preserve their measured transverse separation in this solve;
    # a single-ribbon merge is a separate topology hypothesis, never smoothing.
    from scipy.spatial import cKDTree
    separation_constraints = 0
    for i, j in cKDTree(original).query_pairs(12):
        if vertex_edges[i] & vertex_edges[j]:
            continue
        vi, vj = vertex_tangents[i], vertex_tangents[j]
        delta = original[i]-original[j]
        if abs(vi@vj) < .92 or abs(delta@vi) > 4:
            continue
        normal = np.array([-vi[1], vi[0]])
        separation = delta@normal
        if abs(separation) < .5:
            continue
        add([(2*i, normal[0]), (2*i+1, normal[1]), (2*j, -normal[0]), (2*j+1, -normal[1])], separation, 12.)
        separation_constraints += 1
    dominant = []
    for e in edges:
        simple = np.asarray(e.axis.simplify(3).coords)
        for a, b in zip(simple[:-1], simple[1:]):
            v = b-a; length = np.linalg.norm(v)
            if length > 35:
                dominant.append((np.arctan2(v[1], v[0]) % np.pi, length))
    bins = np.linspace(0, np.pi, 91)
    hist, _ = np.histogram([a for a, _ in dominant], bins, weights=[w for _, w in dominant])
    peaks = []
    for j in np.argsort(hist)[::-1]:
        angle = (bins[j]+bins[j+1])/2
        if hist[j] == 0 or any(abs(np.sin(angle-a)) < np.sin(np.deg2rad(16)) for a in peaks):
            continue
        nearby = [(a, w) for a, w in dominant if abs(np.sin(a-angle)) < np.sin(np.deg2rad(6))]
        angle = .5*np.arctan2(sum(w*np.sin(2*a) for a, w in nearby), sum(w*np.cos(2*a) for a, w in nearby))
        peaks.append(angle)
        if len(peaks) == 4:
            break
    for e, ids in zip(edges, paths):
        xy = original[ids]
        # Second derivative removes pixel zigzags throughout every street.
        for j in range(1, len(ids)-1):
            a, b = xy[j]-xy[j-1], xy[j+1]-xy[j]
            angle = np.arccos(np.clip(a@b/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-8), -1, 1))
            weight = 3. if angle < .35 else .6  # preserve genuine corners/curves
            for d in (0, 1):
                add([(2*ids[j-1]+d, 1), (2*ids[j]+d, -2), (2*ids[j+1]+d, 1)], 0, weight)
        # 40 m windows distinguish a straight road from a curved/roundabout arm.
        for j in range(len(ids)-5):
            q = xy[j:j+6]; v = q[-1]-q[0]; length = np.linalg.norm(v)
            if length < 25:
                continue
            v /= length; n = np.array([-v[1], v[0]])
            if np.max(abs((q-q[0])@n)) > 2.5:
                continue
            angle = np.arctan2(v[1], v[0])
            nearest = min(peaks, key=lambda a: abs(np.sin(a-angle))) if peaks else angle
            if abs(np.sin(nearest-angle)) < np.sin(np.deg2rad(7)):
                n = np.array([-np.sin(nearest), np.cos(nearest)])
            for k in ids[j+1:j+6]:
                add([(2*k, n[0]), (2*k+1, n[1]), (2*ids[j], -n[0]), (2*ids[j]+1, -n[1])], 0, 1.8)
    matrix = coo_matrix((vv, (rr, cc)), shape=(len(values), 2*len(original))).tocsr()
    # Positive positional priors make the normal system nonsingular. A direct
    # sparse solve avoids thousands of badly scaled pinned-node LSQR iterations.
    solved = spsolve((matrix.T@matrix).tocsc(), matrix.T@np.array(values)).reshape(-1, 2)+origin
    solved[list(pinned)] = original[list(pinned)]
    # Entire network uses common displacements, so neighboring edges cannot
    # disagree about the new position of their shared junction.
    delta = solved-original
    limits = np.maximum(1., np.linalg.norm(delta, axis=1)/6.)
    solved = original+delta/limits[:, None]
    rows = []
    numeric_fragments = 0
    for i, (e, ids) in enumerate(zip(edges, paths)):
        line = LineString(solved[ids]).simplify(.5) if i not in locked and e.axis.length >= 16 and not e.axis.is_ring and roi.buffer(-3).covers(e.axis) else e.axis
        # GEOS noding can produce 1e-6..1e-5 metre fragments between the same
        # physical junction. Canonical 0.1 mm coordinates remove that numerical
        # artifact; this is not a short-road deletion threshold.
        coordinates = []
        for point in line.coords:
            point = tuple(np.round(point, 4)) if roi.buffer(-.1).covers(Point(point)) else point
            if not coordinates or point != coordinates[-1]:
                coordinates.append(point)
        if len(coordinates) < 2:
            numeric_fragments += 1
            continue
        line = LineString(coordinates)
        rows.append(dict(frame.iloc[e.sources[0]], geometry=line, block_edge_index=i, repair_role='block_joint_fit'))
    result = gpd.GeoDataFrame(rows, geometry='geometry', crs=frame.crs)
    return result, dict(variables=len(original), equations=len(values), numerical_fragments_collapsed=numeric_fragments, coordinate_precision_m=.0001, separation_constraints=separation_constraints, shared_junctions=sum(len(v) >= 3 for v in nodes.values()),
                        dominant_directions_deg=np.rad2deg(peaks).tolist(),
                        displacement_mean_m=float(np.linalg.norm(delta, axis=1).mean()),
                        displacement_max_m=float(np.linalg.norm(solved-original, axis=1).max()))


def optimize(frame, evidence, roi, config):
    evidence.multichannel_ribbon = True
    evidence.independent_flanks = True
    print('  block paved-patch graph consolidation', flush=True)
    result, audit = consolidate_junctions(frame, evidence, roi)
    print('  completing street-scale interruptions', flush=True)
    result, continuations = complete_streets(result, evidence, roi)
    audit.extend(continuations)
    print('  joint street/junction/direction fitting', flush=True)
    locked = set()
    from graph import candidates
    for attempt in range(4):
        proposed, metrics = joint_fit(result, evidence, roi, locked)
        guard = validate(result, proposed, roi, evidence)
        audit.append(dict(action='block_joint_fit', applied=guard['passed'], attempt=attempt,
                          old_wkt=union_all(result.geometry).wkt, new_wkt=union_all(proposed.geometry).wkt,
                          evidence=dict(metrics, locked_alias_edges=sorted(locked)), transaction_guard=guard,
                          reason='shared_junction_sparse_optimization_rgb_direction_and_curvature'))
        print('  joint fit constraints:', guard['failures'], flush=True)
        if guard['passed']:
            result = proposed
            break
        # Add hard constraints at unresolved parallel aliases and solve the
        # block again. Do not relax the duplicate protection to force acceptance.
        new_edges = build(proposed)
        problem = set()
        if 'duplicate_length_increased' in guard['failures']:
            problem |= {source for c in candidates(new_edges, roi, {}, ('duplicate',))
                        for k in c.ids for source in new_edges[k].sources}
        if 'invalid_or_self_intersecting_geometry' in guard['failures']:
            problem |= {i for i, g in enumerate(proposed.geometry) if not g.is_simple or not g.is_valid or g.length < .001}
        for point in guard['unexpected_crossings']+guard['new_internal_tips']:
            problem |= set(np.flatnonzero(proposed.distance(Point(point)) < .1))
        if 'new_small_loop' in guard['failures']:
            for ring in polygonize(union_all(proposed.geometry)):
                if roi.covers(ring) and ring.length <= 60:
                    problem |= set(np.flatnonzero(proposed.intersects(ring.boundary)))
        problem = {int(proposed.iloc[i].block_edge_index) for i in problem}
        if problem <= locked:
            break
        locked |= problem
    if hasattr(evidence, 'debug_output'):
        proposed.to_file(evidence.debug_output/'block_proposal.gpkg', layer='centerlines', driver='GPKG')
    return result, audit, [dict(stage='whole_block_joint_fit', accepted=guard['passed'], failures=guard['failures'])]

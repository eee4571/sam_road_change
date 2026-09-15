"""Continuous raw-width road surfaces; fine observations remain untouched."""
import geopandas as gpd
import numpy as np
import time
from shapely import make_valid, union_all, line_interpolate_point, STRtree
from shapely.geometry import Polygon, LineString, Point
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


def _polygons(geometry):
    if geometry.geom_type == 'Polygon':
        if not geometry.is_empty:
            yield geometry
    elif hasattr(geometry, 'geoms'):
        for part in geometry.geoms:
            yield from _polygons(part)


def _profile_runs(frame):
    """Rejoin ordered interval axes, not their measurement-cell polygons."""
    for _, group in frame.groupby('parent_id', sort=False):
        if 'segment_id' in group:
            group = group.sort_values('segment_id', kind='stable')
        xy, ss, left, right = [], [], [], []
        length = 0.
        for row in group.itertuples():
            line = row.geometry
            valid = (line.geom_type == 'LineString' and line.length > 1e-7 and
                     np.isfinite([row.final_left_distance, row.final_right_distance]).all() and
                     min(row.final_left_distance, row.final_right_distance) > 0)
            coords = list(line.coords) if valid else []
            connected = coords and xy and np.linalg.norm(np.subtract(xy[-1], coords[0])) < .01
            if xy and not connected:
                yield LineString(xy), np.array(ss), np.array(left), np.array(right)
                xy, ss, left, right, length = [], [], [], [], 0.
            if not valid:
                continue
            xy.extend(coords[1:] if xy else coords)
            ss.append(length + line.length / 2)
            length += line.length
            left.append(row.final_left_distance)
            right.append(row.final_right_distance)
        if xy:
            yield LineString(xy), np.array(ss), np.array(left), np.array(right)


def _display_boundaries(axis, positions, left, right, pixel_size):
    """Stable node-to-node pavement width; measured arrays are read-only."""
    closed = axis.is_ring
    count = max(3, int(np.ceil(axis.length / max(.5, min(1., pixel_size)))))
    distances = np.linspace(0., axis.length, count + 1)
    step = axis.length / count
    xy = np.array(
        [point.coords[0] for point in line_interpolate_point(axis, distances)])
    if closed:
        distances, xy = distances[:-1], xy[:-1]
    mode = 'wrap' if closed else 'nearest'
    # A node-to-node road chain is one presentation-width unit. Sampling-scale
    # width variation belongs in the measured profile, not the pavement outline.
    # A length-weighted median avoids over-weighting short source intervals.
    bounds = np.r_[0., (positions[:-1] + positions[1:]) / 2, axis.length]
    weights = np.maximum(np.diff(bounds), 0.)
    side_widths = []
    for values in (left, right):
        order = np.argsort(values, kind='stable')
        cumulative = np.cumsum(weights[order])
        index = min(len(order)-1, int(np.searchsorted(cumulative, cumulative[-1] / 2)))
        side_widths.append(np.full(len(distances), values[order[index]], dtype=float))
    smooth_xy = gaussian_filter1d(xy, min(4., axis.length / 10) / step, axis=0, mode=mode)
    if not closed:
        # Keep authoritative endpoints and flat cross-sectional terminal caps.
        taper = np.minimum(1., np.minimum(distances, axis.length - distances) / max(1., 3*step))
        smooth_xy = xy + (smooth_xy - xy) * taper[:, None]
    tangent = (np.roll(smooth_xy, -1, axis=0)-np.roll(smooth_xy, 1, axis=0)
               if closed else np.gradient(smooth_xy, axis=0))
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    lxy = smooth_xy + normal * side_widths[0][:, None]
    rxy = smooth_xy - normal * side_widths[1][:, None]
    if closed:
        # Independent closed offset rings avoid a seam across an annular road.
        polygon = make_valid(Polygon(lxy)).symmetric_difference(make_valid(Polygon(rxy)))
    else:
        polygon = make_valid(Polygon(np.vstack([lxy, rxy[::-1]])))
    ends = [(tuple(xy[i]), tangent[i] * (1 if i == 0 else -1),
             float(side_widths[0][i] + side_widths[1][i])) for i in (0, -1)] if not closed else []
    return polygon, ends


def _junction_zones(ends, pixel_size):
    if not ends:
        return []
    points = np.array([entry[0] for entry in ends])
    tree = cKDTree(points)
    visited = set()
    zones = []
    for i, point in enumerate(points):
        if i in visited:
            continue
        ids = tree.query_ball_point(point, max(.05, min(.25, pixel_size / 2)))
        visited.update(ids)
        if len(ids) < 2:
            continue
        if len(ids) == 2 and np.dot(ends[ids[0]][1], ends[ids[1]][1]) < -.92:
            continue
        width = float(np.median([ends[j][2] for j in ids]))
        center = Point(np.mean(points[ids], axis=0))
        rounding = min(4., max(.4, .18 * width))
        zones.append((center.buffer(max(4., width * 1.5)), rounding))
    return zones


def _junction_surfaces(merged, zones, pixel_size):
    """Round only connected junction footprints; never join separate components.

    Processing each pre-existing connected component separately, with expansion
    disabled near other components, protects parallel carriageways. Long/larger
    holes are restored after filleting, including medians through junctions.
    """
    components = list(_polygons(merged))
    component_tree = STRtree(components)
    zone_tree = STRtree([z[0] for z in zones])
    result = []
    for index, component in enumerate(components):
        patches, masks = [], []
        holes = [Polygon(ring) for ring in component.interiors]
        hole_tree = STRtree(holes)
        for raw in zone_tree.query(component, predicate='intersects'):
            zone, radius = zones[int(raw)]
            # The padded cut stays well outside the final patch. Its temporary
            # circular clipping edge can never become a road boundary.
            local = component.intersection(zone.buffer(4 * radius))
            closed = local.buffer(radius).buffer(-radius)
            fillet = closed.buffer(-radius).buffer(radius)
            if len(list(_polygons(fillet))) > len(list(_polygons(local))):
                fillet = closed  # Do not sever a narrow but connected arm.
            protected = []
            for hole_id in sorted(hole_tree.query(zone, predicate='intersects')):
                hole = holes[int(hole_id)]
                # Preserve substantial islands and long medians extending out
                # of the junction. A fully enclosed sub-resolution slit is an
                # internal seam, not a reason to retain a hole after filleting.
                if (not zone.covers(hole) or hole.area > 40 * radius**2 or
                        not hole.buffer(-max(pixel_size, .35 * radius)).is_empty):
                    protected.append(hole)
            if protected:
                fillet = fillet.difference(union_all(protected))
            other_ids = [int(j) for j in component_tree.query(zone.buffer(radius), predicate='intersects') if int(j) != index]
            if other_ids:
                # No expansion of this component where it might close a median.
                fillet = fillet.intersection(component)
            patches.append(fillet.intersection(zone))
            masks.append(zone)
        if patches:
            component = make_valid(union_all([component.difference(union_all(masks)), *patches]))
        result.extend(_polygons(component))
    return union_all(result)


def image_pixel_size(image_path, metric_crs):
    """Physical pixel size without reading or resampling image pixels."""
    import rasterio
    from pyproj import Transformer
    with rasterio.open(image_path) as ds:
        xy = np.array([ds.transform * p for p in [(ds.width/2, ds.height/2),
                       (ds.width/2+1, ds.height/2), (ds.width/2, ds.height/2+1)]])
        x, y = Transformer.from_crs(ds.crs, metric_crs, always_xy=True).transform(xy[:, 0], xy[:, 1])
    points = np.column_stack([x, y])
    return float(min(np.linalg.norm(points[1:] - points[0], axis=1)))


def build_road_surfaces(width_segments, *, pixel_size=None, image_path=None):
    """Presentation surfaces from continuous axes and asymmetric width profiles.

    The input is the fine *line* observations, never corridor polygons. Width
    data used by Fast2 is not mutated; smoothing is exclusive to this product.
    """
    started = time.perf_counter()
    if width_segments.empty:
        return gpd.GeoDataFrame({'surface_id': [], 'area_m2': []}, geometry=[], crs=width_segments.crs)
    metric_crs = (width_segments.estimate_utm_crs() if width_segments.crs.is_geographic or
                  width_segments.crs.axis_info[0].unit_conversion_factor != 1 else width_segments.crs)
    frame = width_segments.to_crs(metric_crs)
    if pixel_size is None:
        if image_path is None:
            raise ValueError('Raw road surface cleanup requires an image pixel size')
        pixel_size = image_pixel_size(image_path, metric_crs)
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError('Invalid road surface pixel size')
    chains, ends, axes, road_widths = [], [], [], []
    for axis, ss, left, right in _profile_runs(frame):
        polygon, terminals = _display_boundaries(axis, ss, left, right, pixel_size)
        chains.append(polygon)
        axes.append(axis)
        road_widths.append(float(np.median(left + right)))
        ends.extend(terminals)
    boundary_seconds = time.perf_counter() - started
    merged = make_valid(union_all(chains))
    print(f'[Raw width] boundary union components={len(list(_polygons(merged)))}', flush=True)
    zones = _junction_zones(ends, pixel_size)
    merged = make_valid(_junction_surfaces(merged, zones, pixel_size))
    rows = []
    axes_tree = STRtree(axes)
    discarded_fragments = 0
    for polygon in _polygons(merged):
        holes = [ring for ring in polygon.interiors if Polygon(ring).area > (pixel_size / 4)**2]
        cleaned = make_valid(Polygon(polygon.exterior, holes).simplify(pixel_size / 20, preserve_topology=True))
        for part in _polygons(cleaned):
            # Do not remove small standalone roads; only numerical zero-area
            # remnants from invalid offset rings lack a meaningful footprint.
            if part.area <= pixel_size**2 * 1e-6:
                continue
            if len(axes_tree.query(part, predicate='intersects')) == 0:
                width = road_widths[int(axes_tree.nearest(part))]
                if part.area < max(pixel_size**2, .25 * width**2):
                    discarded_fragments += 1
                    continue
            rows.append({'surface_id': f'RRS{len(rows):07d}', 'area_m2': part.area, 'geometry': part})
    result = gpd.GeoDataFrame(rows, columns=['surface_id', 'area_m2', 'geometry'], geometry='geometry', crs=metric_crs).to_crs(width_segments.crs)
    # Geographic reprojection can collapse nearly coincident vertices. Validate
    # in the actual exported coordinate system, not just in metric work space.
    result.geometry = make_valid(result.geometry.values)
    result = result.explode(ignore_index=True)
    result = result.loc[result.geom_type.eq('Polygon') & ~result.is_empty].copy()
    result['area_m2'] = result.to_crs(metric_crs).area.values
    result = result.loc[result.area_m2 > pixel_size**2 * 1e-6].reset_index(drop=True)
    result['surface_id'] = [f'RRS{i:07d}' for i in range(len(result))]
    print(f'[Raw width] presentation surfaces: {len(chains)} chains, {len(zones)} junctions -> {len(result)} polygons; '
          f'unsupported_fragments={discarded_fragments} boundary_rebuild={boundary_seconds:.6f}s surface_total={time.perf_counter()-started:.6f}s', flush=True)
    return result

"""Keep polygon truth authoritative while associating it with road intervals.

Voronoi cells assign existing surface to nearby axis samples; they never create
road area. Every result is an intersection with the supplied footprint. Axes
are association metadata, not instructions to regenerate polygon boundaries.
"""
import numpy as np
from shapely import union_all, voronoi_polygons
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.strtree import STRtree


def _sites(axis, extra=()):
    # Sampling controls association only. Original boundary vertices and holes
    # are preserved even for very long roads; no raster is allocated.
    count = min(100_000, max(2, int(np.ceil(axis.length / 2.)) + 1))
    distances = np.unique(np.r_[np.linspace(0., axis.length, count), extra])
    return distances, np.array([axis.interpolate(s).coords[0][:2] for s in distances])


def _cells(surface, coordinates):
    unique, indices = np.unique(coordinates, axis=0, return_index=True)
    if len(unique) == 1:
        yield int(indices[0]), surface
        return
    points = MultiPoint(unique)
    tree = STRtree(list(points.geoms))
    cells = voronoi_polygons(points, extend_to=surface.envelope)
    for cell in cells.geoms:
        # A Voronoi cell belongs to the nearest generator at an interior point.
        # Almost coincident endpoint/sample coordinates can both be covered by
        # a cell after GEOS rounding, so containment is not a unique owner test.
        # Resolve exact distance ties by input order for reproducible assignment.
        owners = tree.query_nearest(cell.representative_point(), all_matches=True)
        owner = min(int(indices[int(index)]) for index in owners)
        clipped = cell.intersection(surface)
        if not clipped.is_empty and clipped.area > 0:
            yield owner, clipped


def partition_surface(surface, axes):
    """Disjoint branch footprints whose union preserves the entire input face."""
    if len(axes) == 1:
        return [surface]
    coordinates, owners = [], []
    for index, axis in enumerate(axes):
        _, xy = _sites(axis)
        coordinates.extend(xy)
        owners.extend([index] * len(xy))
    parts = [[] for _ in axes]
    for site, geometry in _cells(surface, np.asarray(coordinates)):
        parts[owners[site]].append(geometry)
    return [union_all(group) if group else Polygon() for group in parts]


def surface_interval(surface, reference, local):
    """Select a longitudinal subset without buffering or widening the face.

    Endpoint cells are cut normal to the local tangent. Applying these cuts only
    in their own cells avoids a cut across an unrelated arm of a U-shaped road.
    """
    if local.is_empty or local.length <= 1e-9:
        return Polygon()
    if local.equals(reference):
        return surface
    start, end = sorted(reference.project(Point(point))
                        for point in (local.coords[0], local.coords[-1]))
    if start <= 1e-8 and end >= reference.length - 1e-8:
        return surface
    stations, coordinates = _sites(reference, (start, end))
    span = max(np.ptp(np.asarray(surface.bounds).reshape(2, 2), axis=0).max(), reference.length, 1.) * 4

    def half_plane(station, forward):
        center = np.asarray(reference.interpolate(station).coords[0][:2])
        lo, hi = max(0., station-.1), min(reference.length, station+.1)
        tangent = np.asarray(reference.interpolate(hi).coords[0][:2]) - np.asarray(reference.interpolate(lo).coords[0][:2])
        norm = np.linalg.norm(tangent)
        if norm <= 1e-12:
            # At an exact reversal the central difference cancels. Use the
            # one-sided tangent of the retained interval at this endpoint.
            tangent = (np.asarray(reference.interpolate(hi).coords[0][:2]) - center
                       if forward else center - np.asarray(reference.interpolate(lo).coords[0][:2]))
            norm = np.linalg.norm(tangent)
        if norm <= 1e-12:
            # No directional cut is defined for this endpoint. The cell already
            # belongs to the selected interval and lies inside the source face.
            return surface.envelope
        tangent *= (1. if forward else -1.) / norm
        normal = np.array([-tangent[1], tangent[0]])
        return Polygon([center-normal*span, center+normal*span,
                        center+normal*span+tangent*span, center-normal*span+tangent*span])

    selected = []
    for site, cell in _cells(surface, coordinates):
        station = stations[site]
        if not start-1e-9 <= station <= end+1e-9:
            continue
        if abs(station-start) < 1e-9 and start > 1e-8:
            cell = cell.intersection(half_plane(start, True))
        if abs(station-end) < 1e-9 and end < reference.length-1e-8:
            cell = cell.intersection(half_plane(end, False))
        selected.append(cell)
    return union_all(selected)

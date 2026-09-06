"""Render already approved Auto objects from metric axes and final widths only.

No mask, surface, probability, truth, matching decision or acceptance threshold
is available to this renderer. Assembly membership and routes are read-only.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter
from shapely import make_valid, union_all
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

from .auto_change_assembly import polygonal


def _clean_overlay(geometry):
    """Remove sub-decimetre overlay debris, preserving real corridor islands.

    These defects arise from almost coincident offset edges in metre CRS, not
    masks. Never drop an entire seed, or fill the inner space of a road loop.
    """
    geometry = polygonal(geometry)
    parts = [geometry] if geometry.geom_type == 'Polygon' else list(geometry.geoms)
    if not parts:
        return geometry
    largest = max(parts, key=lambda p: p.area)
    return union_all([Polygon(p.exterior, [ring for ring in p.interiors if Polygon(ring).area > .01])
                      for p in parts if p.area > .01 or p is largest])


def _stations(axis):
    return np.linspace(0., axis.length, max(2, int(np.ceil(axis.length / 4.)) + 1))


def _smooth(values):
    # Positive, non-overshooting filters: suppress segment steps without moving
    # the final axis or inventing extrema in a measured width profile.
    return gaussian_filter1d(median_filter(np.asarray(values, float), size=3, mode='nearest'),
                             sigma=1.5, mode='nearest')


def _direction(axis, station):
    a = np.asarray(axis.interpolate(max(0., station - 1.)).coords[0])
    b = np.asarray(axis.interpolate(min(axis.length, station + 1.)).coords[0])
    return (b - a) / max(np.linalg.norm(b - a), 1e-12)


class FinalWidths:
    """Read final width segments on the same axis; never reach across a track."""
    def __init__(self, frame):
        self.frame = frame.explode(index_parts=False).reset_index(drop=True)
        self.tree = STRtree(self.frame.geometry.values)

    def profile(self, axis, fallback):
        stations = _stations(axis)
        values = np.full(len(stations), np.nan)
        for i, station in enumerate(stations):
            point = axis.interpolate(station)
            direction = _direction(axis, station)
            choices = []
            for index in self.tree.query(point, predicate='dwithin', distance=.75):
                row = self.frame.iloc[int(index)]
                line = row.geometry
                if line.geom_type != 'LineString' or line.length <= 0:
                    continue
                cosine = abs(float(direction @ _direction(line, line.project(point))))
                width = float(row.get('width_m', np.nan))
                if cosine >= .95 and np.isfinite(width) and width > 0:
                    choices.append((line.distance(point), -cosine, int(index), width))
            if choices:
                values[i] = min(choices)[-1]
        found = np.isfinite(values)
        values = (np.interp(stations, stations[found], values[found]) if found.any()
                  else np.full(len(stations), float(fallback)))
        if not np.isfinite(values).all() or np.min(values) <= 0:
            raise ValueError('Published road geometry requires positive final widths')
        return stations, _smooth(values), float(found.mean())


def corridor(axis, stations, widths):
    """Flat-ended variable-width corridor on the exact final/canonical axis.

    Original axis vertices are retained. Bounded mitres avoid spikes at bends;
    no polygon smoothing can move the axis, merge tracks or copy mask defects.
    """
    if axis.geom_type != 'LineString' or axis.length <= 0:
        raise ValueError('Published change has no usable longitudinal axis')
    if np.ptp(widths) < 1e-8:
        return polygonal(axis.buffer(float(widths[0])/2, cap_style='flat', join_style='round'))
    original = np.asarray(axis.coords)[:, :2]
    vertex_stations = np.r_[0., np.cumsum(np.linalg.norm(np.diff(original, axis=0), axis=1))]
    sample_stations = np.unique(np.r_[stations, vertex_stations])
    coords = np.asarray([axis.interpolate(s).coords[0] for s in sample_stations])[:, :2]
    keep = np.r_[True, np.linalg.norm(np.diff(coords, axis=0), axis=1) > 1e-8]
    coords, sample_stations = coords[keep], sample_stations[keep]
    directions = np.diff(coords, axis=0)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    normals = np.column_stack([-directions[:, 1], directions[:, 0]])
    offsets = np.vstack([normals[0], normals[:-1] + normals[1:], normals[-1]])
    offsets /= np.maximum(np.linalg.norm(offsets, axis=1)[:, None], 1e-12)
    denominator = np.r_[1., np.sum(offsets[1:-1] * normals[1:], axis=1), 1.]
    offsets /= np.maximum(denominator[:, None], .5)
    offsets *= np.interp(sample_stations, stations, widths)[:, None]/2
    return _clean_overlay(make_valid(Polygon(np.vstack([coords + offsets, (coords - offsets)[::-1]]))))


def _paired_profile(row, axis, samples):
    stations = _stations(axis)
    before = np.full(len(stations), float(row.width_bef))
    after = np.full(len(stations), float(row.width_aft))
    origin = 'approved_paired_run_widths'
    if len(samples):
        local = samples.loc[samples.candidate_id == row.candidate_id].sort_values('station_m')
        sign = 1 if row.change_typ == 'widened' else -1
        # This is rendering of the approved interval, not another qualification.
        # Invalid/contradictory samples do not cut or shorten the object: the
        # approved run widths describe portions without a usable paired profile.
        local = local.loc[local.valid.astype(bool) & (local.before_width > 0) & (local.after_width > 0)
                          & (sign*(local.after_width-local.before_width) > 0)]
        if len(local) >= 2:
            before = _smooth(np.interp(stations, local.station_m, local.before_width))
            after = _smooth(np.interp(stations, local.station_m, local.after_width))
            origin = 'saved_paired_width_profile'
    return stations, before, after, origin


def render_change_geometry(objects, seeds, assembly, final_widths, width_samples=None):
    """Replace polygon geometry only, preserving IDs, QA, membership and paths.

    All input WKT/assembly axes and final_widths use the seeds' metric CRS.
    Returns objects plus explicit regularized seed/bridge and profile diagnostics.
    Missing axes raise an error instead of silently publishing raw geometry.
    """
    metric = seeds.crs
    axes = assembly['object_axes'].to_crs(metric)
    if seeds.empty:
        empty = gpd.GeoDataFrame(geometry=[], crs=metric)
        return objects.copy(), {'published_geometry_parts': empty, 'published_width_profiles': empty.copy()}
    widths = {side: FinalWidths(frame.to_crs(metric)) for side, frame in final_widths.items()}
    samples = pd.DataFrame() if width_samples is None else pd.DataFrame(width_samples)
    profiles, pieces, profile_rows = {}, [], []
    seed_axes = {int(row.seed_id): row.geometry for row in axes.itertuples() if row.role == 'seed'}
    if set(seed_axes) != set(range(len(seeds))):
        raise ValueError('Cannot render a published seed without its saved final/canonical axis')

    def add_piece(axis, stations, before, after, kind, role, object_id, seed_id, origin, coverage):
        if kind in ('added', 'removed'):
            geometry = corridor(axis, stations, after if kind == 'added' else before)
        else:
            b, a = corridor(axis, stations, before), corridor(axis, stations, after)
            geometry = _clean_overlay(a.difference(b) if kind == 'widened' else b.difference(a))
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError(f'Empty/invalid regularized geometry: {object_id}/{seed_id}/{role}')
        pieces.append(dict(object_id=object_id, seed_id=seed_id, change_typ=kind, role=role,
                           geometry_source=origin, final_width_coverage=coverage, geometry=geometry))
        for s, b, a in zip(stations, before, after):
            profile_rows.append(dict(object_id=object_id, seed_id=seed_id, role=role, station_m=s,
                                     width_bef=b, width_aft=a, geometry=axis.interpolate(s)))

    membership = assembly['membership'].set_index('seed_id') if len(seeds) else pd.DataFrame()
    for seed_id, row in seeds.reset_index(drop=True).iterrows():
        axis = seed_axes[seed_id]
        coverage = 1.
        if row.change_typ in ('added', 'removed'):
            side = 'after' if row.change_typ == 'added' else 'before'
            stations, values, coverage = widths[side].profile(axis, row.width_aft if side == 'after' else row.width_bef)
            before, after = (np.zeros(len(values)), values) if side == 'after' else (values, np.zeros(len(values)))
            origin = f'{side}_final_axis_and_width'
        else:
            stations, before, after, origin = _paired_profile(row, axis, samples)
        profiles[seed_id] = axis, stations, before, after
        add_piece(axis, stations, before, after, row.change_typ, 'seed', membership.loc[seed_id, 'object_id'],
                  seed_id, origin, coverage)

    # object_axes preserves the assembly bridge order within each object. The
    # endpoint IDs come from the frozen assembly, never geometric proximity.
    bridges = assembly['assembly_bridges']
    for object_id, local_axes in axes.loc[axes.role != 'seed'].groupby('object_id', sort=False):
        ids = set(assembly['membership'].loc[assembly['membership'].object_id == object_id, 'seed_id'])
        local_bridges = bridges.loc[bridges.first_seed.isin(ids)]
        if len(local_axes) != len(local_bridges):
            raise ValueError('Assembly bridge paths and membership disagree')
        for (_, axis_row), (_, bridge) in zip(local_axes.iterrows(), local_bridges.iterrows()):
            axis = axis_row.geometry
            if axis.length <= 1e-8:
                # Coincident seed endpoints are already connected. A zero-length
                # square buffer would introduce an artificial junction bulge.
                pieces.append(dict(object_id=object_id, seed_id=-1, change_typ=axis_row.change_typ,
                    role=axis_row.role, geometry_source='coincident_endpoint_no_extra_area',
                    final_width_coverage=1., geometry=Polygon()))
                continue
            stations = _stations(axis)
            ends = []
            for seed_id, endpoint in ((int(bridge.first_seed), 0), (int(bridge.second_seed), -1)):
                seed_axis, ss, b, a = profiles[seed_id]
                s = seed_axis.project(Point(axis.coords[endpoint]))
                ends.append((np.interp(s, ss, b), np.interp(s, ss, a)))
            before = np.interp(stations, [0, axis.length], [ends[0][0], ends[1][0]])
            after = np.interp(stations, [0, axis.length], [ends[0][1], ends[1][1]])
            add_piece(axis, stations, before, after, axis_row.change_typ, axis_row.role, object_id, -1,
                      'frozen_bridge_axis_and_endpoint_widths', 1.)

    def frame(rows):
        return (gpd.GeoDataFrame(rows, geometry='geometry', crs=metric) if rows else
                gpd.GeoDataFrame(geometry=[], crs=metric))
    parts = frame(pieces)
    result = objects.to_crs(metric).copy()
    for index, row in result.iterrows():
        result.at[index, 'geometry'] = _clean_overlay(union_all(parts.loc[parts.object_id == row.object_id].geometry.values))
    result = result.to_crs(objects.crs)
    return result, {'published_geometry_parts': parts, 'published_width_profiles': frame(profile_rows)}

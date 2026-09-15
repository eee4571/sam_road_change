"""Standalone raw RGB boundary experiment. No project algorithm imports."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import LineString


@dataclass
class Config:
    spacing: float = 3.0
    smooth_sigma: float = 2.0
    tangent_span: float = 5.0
    cross_step: float = 0.5
    search_radius: float = 20.0
    min_side: float = 0.75
    min_width: float = 2.0
    max_width: float = 36.0
    candidates: int = 7
    junction_radius: float = 12.0
    confidence_threshold: float = 0.16
    position_weight: float = 0.35
    width_weight: float = 0.50
    min_length: float = 70.0
    max_roads: int = 120
    min_run_samples: int = 3


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_roads(path, layer, config):
    source = gpd.read_file(path, layer=layer)
    if source.crs is None:
        raise ValueError('Centerlines require an explicit CRS')
    metric_crs = source.estimate_utm_crs()
    source = source.to_crs(metric_crs)
    # Union deduplicates reverse predictions and nodes actual intersections.
    network = shapely.union_all(source.geometry)
    parts = list(shapely.get_parts(network))
    degree = {}
    for line in parts:
        if line.geom_type != 'LineString':
            raise ValueError('Only line input is supported')
        for xy in (line.coords[0], line.coords[-1]):
            key = tuple(np.round(xy[:2], 5))
            degree[key] = degree.get(key, 0) + 1
    junctions = np.array([xy for xy, n in degree.items() if n >= 3]).reshape(-1, 2)
    merged = list(shapely.get_parts(shapely.line_merge(network)))
    merged = sorted((l for l in merged if l.length > 0 and l.length >= config.min_length),
                    key=lambda l: (-l.length, l.wkt))
    total = len(merged)
    if config.max_roads:
        merged = merged[:config.max_roads]
    return merged, junctions, metric_crs, dict(input_features=len(source), eligible_roads=total,
        selection=('all nonzero merged degree-2 chains meeting min_length' if not config.max_roads else
                   'longest merged degree-2 chains; deterministic, not representative random sample'))


def sample_line(line, c):
    # Smooth at 1 m resolution, preserve the two network endpoints, then resample by arc length.
    s = np.linspace(0, line.length, max(3, int(np.ceil(line.length)) + 1))
    xy = shapely.get_coordinates(shapely.line_interpolate_point(line, s))
    smooth = gaussian_filter1d(xy, c.smooth_sigma / (s[1] - s[0]), axis=0, mode='nearest')
    smooth[0], smooth[-1] = xy[0], xy[-1]
    smoothed = LineString(smooth)
    distance = np.arange(0, smoothed.length + 1e-8, c.spacing)
    xy = shapely.get_coordinates(shapely.line_interpolate_point(smoothed, distance))
    before = shapely.get_coordinates(shapely.line_interpolate_point(smoothed, np.maximum(0, distance-c.tangent_span)))
    after = shapely.get_coordinates(shapely.line_interpolate_point(smoothed, np.minimum(smoothed.length, distance+c.tangent_span)))
    tangent = after-before
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    return distance, xy, normal


class ImageReader:
    def __init__(self, path, metric_crs):
        self.ds = rasterio.open(path)
        if self.ds.count < 3 or self.ds.dtypes[:3] != ('uint8',)*3:
            raise ValueError('Experiment requires three uint8 RGB bands in order 1,2,3')
        if self.ds.crs is None:
            raise ValueError('Image requires an explicit CRS')
        self.to_image = Transformer.from_crs(metric_crs, self.ds.crs, always_xy=True)

    def pixels(self, xy):
        x, y = self.to_image.transform(xy[..., 0], xy[..., 1])
        inv = ~self.ds.transform
        col = inv.a*x + inv.b*y + inv.c - 0.5
        row = inv.d*x + inv.e*y + inv.f - 0.5
        return np.stack([col, row], axis=-1)

    def crop(self, xy, pad=5):
        pix = self.pixels(xy)
        lo = np.maximum(np.floor(pix.reshape(-1, 2).min(0)-pad), 0).astype(int)
        hi = np.minimum(np.ceil(pix.reshape(-1, 2).max(0)+pad+1), [self.ds.width, self.ds.height]).astype(int)
        if np.any(hi <= lo):
            raise ValueError('Centerline does not overlap image')
        window = Window(*lo, *(hi-lo))
        rgb = self.ds.read([1, 2, 3], window=window).transpose(1, 2, 0)
        valid = (self.ds.read_masks([1, 2, 3], window=window) > 0).all(0)
        return rgb, valid, pix-lo, lo


def normalize_feature(a, scale):
    return np.clip(a / scale, 0, 1)


def extract_profiles(reader, xy, normal, c):
    # Read an evidence halo so a boundary at the search limit still has both color bands.
    radius = c.search_radius + 2.0
    offsets = np.arange(-radius, radius+c.cross_step/2, c.cross_step)
    points = xy[:, None, :] + normal[:, None, :]*offsets[None, :, None]
    rgb, mask, pix, _ = reader.crop(points)
    rgbf = rgb.astype(np.float32)/255
    lab = cv2.cvtColor(rgbf, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(rgbf, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 0.8)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)/8
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)/8
    edge = np.hypot(gx, gy)
    canny = cv2.dilate(cv2.Canny((blurred*255).astype('uint8'), 40, 100), np.ones((3, 3), 'uint8'))/255
    variance = cv2.boxFilter(gray*gray, -1, (7, 7))-cv2.boxFilter(gray, -1, (7, 7))**2
    texture = np.sqrt(np.maximum(variance, 0))
    coordinates = [pix[..., 1], pix[..., 0]]
    def sample(a, order=1, fill=np.nan):
        return map_coordinates(a, coordinates, order=order, mode='constant', cval=fill)
    valid = sample(cv2.erode(mask.astype('uint8'), np.ones((3, 3), 'uint8')).astype(float), fill=0) > .99
    rgb_profile = np.stack([sample(rgbf[..., k]) for k in range(3)], -1)
    lp = np.stack([sample(lab[..., k]) for k in range(3)], -1)
    lp = np.nan_to_num(lp)
    # Compare bands 0.5-2 m on either side of a candidate; no assumption that center is asphalt.
    def contrast(a):
        if a.ndim == 2:
            a = a[..., None]
        k = max(1, round(2/c.cross_step))
        padded = np.pad(a, ((0, 0), (k, k), (0, 0)), mode='edge')
        minus = sum(padded[:, k-j:k-j+a.shape[1]] for j in range(1, k+1))/k
        plus = sum(padded[:, k+j:k+j+a.shape[1]] for j in range(1, k+1))/k
        return np.linalg.norm(plus-minus, axis=-1)
    color = normalize_feature(contrast(lp), 22)
    gradient = normalize_feature(np.linalg.norm(np.gradient(lp, c.cross_step, axis=1), axis=-1), 12)
    tex = normalize_feature(contrast(np.nan_to_num(sample(texture))), .07)
    edges = .65*normalize_feature(np.nan_to_num(sample(edge)), .10) + .35*np.nan_to_num(sample(canny))
    features = np.stack([color, gradient, tex, edges], axis=-1)
    score = features @ np.array([.35, .25, .15, .25])
    # Require all evidence support pixels within +/-2 m to be valid.
    k = max(1, round(2/c.cross_step))
    support = cv2.erode(valid.astype('uint8'), np.ones((1, 2*k+1), 'uint8'),
                        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    score[~support] = -1
    return offsets, score, features, valid, rgb_profile, lp


def side_candidates(offsets, scores, sign, c):
    ids = np.where((sign*offsets >= c.min_side) & (sign*offsets <= c.search_radius))[0]
    response = scores[ids]
    peaks, _ = find_peaks(response, distance=max(1, round(1/c.cross_step)))
    # Endpoints are admissible, but search-limit winners will be flagged.
    pool = np.unique(np.r_[peaks, 0, len(ids)-1])
    pool = pool[response[pool] >= 0]
    pool = pool[np.argsort(-response[pool], kind='stable')[:c.candidates]]
    return ids[pool]


def states_for_row(offsets, score, c):
    left = side_candidates(offsets, score, 1, c)
    right = side_candidates(offsets, score, -1, c)
    if not len(left) or not len(right):
        return np.empty((0, 2), int), np.empty((0, 2)), np.empty(0)
    ids = np.array(np.meshgrid(left, right)).reshape(2, -1).T
    distances = np.column_stack([offsets[ids[:, 0]], -offsets[ids[:, 1]]])
    widths = distances.sum(1)
    keep = (widths >= c.min_width) & (widths <= c.max_width)
    ids, distances = ids[keep], distances[keep]
    return ids, distances, score[ids].sum(1)


def huber(x, delta=1.0):
    a = np.abs(x)
    return np.where(a <= delta, .5*a*a, delta*(a-.5*delta))


def optimize_chain(states, spacing, c):
    """Exact Viterbi solution on joint (left,right) candidate states; no symmetry prior."""
    costs = -states[0][2]
    back = []
    for previous, current in zip(states[:-1], states[1:]):
        old, new = previous[1], current[1]
        delta = (new[:, None, :]-old[None, :, :])/spacing
        width_delta = (new.sum(1)[:, None]-old.sum(1)[None, :])/spacing
        transition = c.position_weight*huber(delta).sum(2) + c.width_weight*huber(width_delta)
        total = transition + costs[None, :]
        parent = total.argmin(1)
        costs = total[np.arange(len(new)), parent]-current[2]
        back.append(parent)
    selected = [int(costs.argmin())]
    for parent in reversed(back):
        selected.append(int(parent[selected[-1]]))
    return np.array(selected[::-1])


def runs(mask):
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.where(changes == 1)[0], np.where(changes == -1)[0])


def confidence_for_choice(ids, score, offsets):
    result = []
    for idx in ids:
        same_side = offsets*offsets[idx] > 0
        competitors = score[same_side & (np.abs(offsets-offsets[idx]) >= 2)]
        competitor = max(0., float(competitors.max())) if len(competitors) else 0.
        strength = max(0., float(score[idx]))
        # Ambiguous repeated edges retain low confidence even if locally strong.
        uniqueness = np.clip((strength-competitor)/.3, 0, 1)
        result.append(strength*(.25+.75*uniqueness))
    return np.array(result)


def measure_road(reader, line, junction_tree, c):
    distance, xy, normal = sample_line(line, c)
    offsets, scores, features, valid, rgb_profile, lab_profile = extract_profiles(reader, xy, normal, c)
    states = [states_for_row(offsets, score, c) for score in scores]
    baseline = np.array([int(s[2].argmax()) if len(s[2]) else -1 for s in states])
    chosen = baseline.copy()
    flags = [set() for _ in xy]
    near_junction = junction_tree.query(xy)[0] < c.junction_radius if junction_tree is not None else np.zeros(len(xy), bool)
    for i, (s, index) in enumerate(zip(states, baseline)):
        if near_junction[i]:
            flags[i].add('junction')
        if not valid[i].all():
            flags[i].add('image_or_nodata_boundary')
        if index < 0:
            flags[i].add('no_valid_pair')
            continue
        conf = confidence_for_choice(s[0][index], scores[i], offsets)
        if conf.min() < c.confidence_threshold:
            flags[i].add('ambiguous_or_weak')
        if s[1][index].max() >= c.search_radius-c.cross_step:
            flags[i].add('search_limit')
    # Fixed-point splitting: selected weak states are removed from chains and never bridge a gap.
    for _ in range(len(xy)+1):
        good = np.array([not f for f in flags])
        chosen = baseline.copy()
        for a, b in runs(good):
            chosen[a:b] = optimize_chain(states[a:b], c.spacing, c)
        newly_bad = False
        for i in np.flatnonzero(good):
            ids = states[i][0][chosen[i]]
            if confidence_for_choice(ids, scores[i], offsets).min() < c.confidence_threshold:
                flags[i].add('optimized_weak'); newly_bad = True
            if states[i][1][chosen[i]].max() >= c.search_radius-c.cross_step:
                flags[i].add('search_limit'); newly_bad = True
        if not newly_bad:
            break
    # A single strong section between rejected sections has no longitudinal support.
    # Exclude short fragments rather than displaying them as a continuous width result.
    for a, b in runs(np.array([not f for f in flags])):
        if b-a < c.min_run_samples:
            for i in range(a, b):
                flags[i].add('insufficient_continuous_support')
                chosen[i] = baseline[i]
    data = dict(s=distance, xy=xy, normal=normal, offsets=offsets, scores=scores,
                features=features, rgb=rgb_profile, lab=lab_profile)
    data['flags'] = [';'.join(sorted(f)) for f in flags]
    for name, selection in [('baseline', baseline), ('optimized', chosen)]:
        widths = np.full((len(xy), 2), np.nan)
        confidence = np.zeros((len(xy), 2))
        response = np.zeros((len(xy), 2))
        for i, index in enumerate(selection):
            if index >= 0:
                widths[i] = states[i][1][index]
                ids = states[i][0][index]
                confidence[i] = confidence_for_choice(ids, scores[i], offsets)
                response[i] = scores[i][ids]
        data[name] = widths
        data[name+'_confidence'] = confidence
        data[name+'_response'] = response
    return data

"""ROI image descriptors and cached width observations; no inference or measurement."""
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from scipy.ndimage import map_coordinates
from shapely import covers, points, union_all
from shapely.geometry import LineString

import bootstrap
from data import read_roi
from engine.road_connection_evidence import ConnectionEvidence, RoadProbability, ArrayRoadProbability
from engine.road_axis_quality import _evidence_route
from engine.fast_image_structure import normalized_gray, gradients


def stations(line, spacing=2.5):
    ss = np.linspace(0, line.length, max(3, int(np.ceil(line.length / spacing)) + 1))
    xy = np.array([line.interpolate(s).coords[0] for s in ss])
    tangent = np.array([np.subtract(line.interpolate(min(line.length, s + 3)).coords[0],
                                   line.interpolate(max(0, s - 3)).coords[0]) for s in ss])
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-9)
    return ss, xy, tangent, np.c_[-tangent[:, 1], tangent[:, 0]]


def longest_hole(supported, length):
    runs = np.diff(np.r_[0, np.flatnonzero(supported) + 1, len(supported) + 1]) - 1
    return min(float(length), float(runs.max(initial=0) * length / max(1, len(supported) - 1)))


class ImageGrid:
    def __init__(self, paths, crs, roi, resolution):
        x0, y0, x1, y1 = roi.bounds
        self.resolution = resolution
        self.transform = from_origin(x0, y1, resolution, resolution)
        w, h = int(np.ceil((x1-x0)/resolution)), int(np.ceil((y1-y0)/resolution))
        if w*h > 20_000_000:
            raise ValueError("ROI exceeds 20 million analysis pixels; use a smaller --bbox or coarser config")
        self.rgb = np.full((h, w, 3), np.nan, dtype=np.float32)
        for path in paths:
            with rasterio.open(path) as ds:
                if ds.crs is None:
                    raise ValueError(f"Image has no CRS: {path}")
                with WarpedVRT(ds, crs=crs, transform=self.transform, width=w, height=h,
                               resampling=Resampling.bilinear, dtype="float32", nodata=np.nan) as vrt:
                    channels = [1, 2, 3] if ds.count >= 3 else [1, 1, 1]
                    block = vrt.read(channels, masked=True).filled(np.nan).transpose(1, 2, 0)
                valid = np.isfinite(block).all(axis=2)
                self.rgb[valid] = block[valid]
        self.valid = np.isfinite(self.rgb).all(axis=2)
        # Existing normalization is an in-memory descriptor, not IR-MAD fitting.
        self.gray, gray_valid = normalized_gray(self.rgb.transpose(2, 0, 1))
        self.valid &= gray_valid
        self.gx, self.gy = gradients(np.nan_to_num(self.gray).astype(np.float32))
        magnitude = np.hypot(self.gx, self.gy)
        self.scale = max(float(np.quantile(magnitude[self.valid], .8)), .01) if self.valid.any() else 1.

    def sample(self, array, xy):
        col, row = (~self.transform) * (xy[:, 0], xy[:, 1])
        value = map_coordinates(array, [row-.5, col-.5], order=1, mode="constant", cval=np.nan)
        valid = map_coordinates(self.valid.astype(float), [row-.5, col-.5], order=1,
                                mode="constant", cval=0) > .999
        return np.where(valid, value, np.nan)

    def cross_sections(self, xy, normal, width):
        offsets = np.linspace(-max(width, 3), max(width, 3), 41)
        probes = xy[:, None, :] + normal[:, None, :] * offsets[None, :, None]
        flat = probes.reshape(-1, 2)
        gx = self.sample(self.gx, flat).reshape(len(xy), -1)
        gy = self.sample(self.gy, flat).reshape(len(xy), -1)
        perpendicular = np.abs(gx*normal[:, 0, None] + gy*normal[:, 1, None])
        parallel = np.abs(-gx*normal[:, 1, None] + gy*normal[:, 0, None])
        energy = perpendicular * perpendicular / (perpendicular + parallel + 1e-6)
        scores = []
        for sign in (-1, 1):
            use = (offsets*sign >= width*.25) & (offsets*sign <= width*.8)
            values = energy[:, use]
            score = np.max(np.where(np.isfinite(values), values, -np.inf), axis=1)
            scores.append(np.where(np.isfinite(score), np.clip(score/self.scale, 0, 1), np.nan))
        return offsets, self.sample(self.gray, flat).reshape(len(xy), -1), scores


class Evidence:
    def __init__(self, inputs, crs, roi, config):
        self.config = config
        self._measure_cache = {}
        self._route_cache = {}
        self.image = ImageGrid(inputs["images"], crs, roi, config["raster_resolution_m"])
        surface_path = inputs["assets"]["surfaces"]
        self.surface = union_all(read_roi(surface_path, crs, roi).geometry) if surface_path.is_file() else None
        # Sample each existing raster once into a bounded in-memory ROI grid.
        h, w = self.image.valid.shape
        rr, cc = np.indices((h, w))
        x, y = self.image.transform * (cc.ravel()+.5, rr.ravel()+.5)
        xy = np.c_[x, y]
        def cached(sources):
            if not sources:
                return None
            reader = RoadProbability(sources, crs)
            values = np.concatenate([reader.values(chunk) for chunk in np.array_split(xy, max(1, len(xy)//100000))])
            return ArrayRoadProbability([dict(mask=values.reshape(h, w), transform=self.image.transform)], crs, crs)
        self.probability = cached(inputs["probabilities"])
        self.molra = cached(inputs["molra"])
        self.base = ConnectionEvidence(self.surface, self.probability, self.molra)
        path = inputs["assets"]["observations"]
        self.observations = read_roi(path, crs, roi, "point_profiles") if path.is_file() else None

    def values(self, xy):
        p = np.full(len(xy), np.nan) if self.probability is None else self.probability.values(xy)
        m = np.full(len(xy), np.nan) if self.molra is None else self.molra.values(xy)
        s = np.zeros(len(xy)) if self.surface is None else covers(self.surface, points(xy)).astype(float)
        return p, m, s

    def measure(self, line, width):
        key = (line.wkb, float(width))
        if key in self._measure_cache:
            return dict(self._measure_cache[key])
        ss, xy, tangent, normal = stations(line, self.config["sample_m"])
        offsets, gray, (left, right) = self.image.cross_sections(xy, normal, width)
        p, m, s = self.values(xy)
        bilateral = np.minimum(left, right)
        independent = (p >= self.config["probability_threshold"]) | (m >= .5)
        supported = independent & (bilateral >= .25)
        result = self.base.measure(line)
        valid = np.isfinite(p) | np.isfinite(m)
        def mean(values):
            return float(np.nanmean(values)) if np.isfinite(values).any() else None
        result.update(left_edge_support=mean(left), right_edge_support=mean(right),
                      bilateral_edge_score=mean(bilateral), rgb_valid_fraction=float(np.isfinite(bilateral).mean()),
                      independent_valid_fraction=float(valid.mean()),
                      independent_support_fraction=float(independent.mean()),
                      continuous_support_fraction=float(supported.mean()),
                      maximum_unsupported_m=longest_hole(supported, line.length),
                      evidence_score=mean(np.where(valid & np.isfinite(bilateral),
                          .65*np.fmax(p, m) + .35*bilateral, np.nan)),
                      cached_width_cv=None, cached_width_quality=None, cached_width_samples=0)
        # Reuse saved point profiles; direction and distance prevent borrowing an adjacent road's widths.
        if self.observations is not None and len(self.observations):
            obs = self.observations
            ids = obs.sindex.query(line.buffer(min(2., width/4)), predicate="intersects")
            subset = obs.iloc[ids]
            if len(subset):
                use = []
                for _, row in subset.iterrows():
                    d = line.project(row.geometry)
                    local = np.array(line.interpolate(min(line.length, d+2)).coords[0])-np.array(line.interpolate(max(0, d-2)).coords[0])
                    local /= max(np.linalg.norm(local), 1e-9)
                    use.append(abs(local @ np.array([row.normal_x, row.normal_y])) < .25)
                subset = subset[np.array(use)]
                widths = subset.final_width.to_numpy(float)
                good = np.isfinite(widths) & (widths > 0)
                if good.any():
                    result.update(cached_width_cv=float(np.std(widths[good])/np.mean(widths[good])),
                                  cached_width_quality=float(subset.final_confidence.to_numpy(float)[good].mean()),
                                  cached_width_samples=int(good.sum()))
                    result["cached_width_observations"] = subset.loc[good, [
                        "road_id", "sample_id", "final_width", "final_confidence", "flags", "width_source"]].to_dict("records")
        result["profile"] = dict(stations_m=ss.tolist(), offsets_m=offsets.tolist(),
                                 gray=gray.tolist(), probability=p.tolist(), molra=m.tolist(),
                                 formal_surface=s.tolist(), left_edge=left.tolist(), right_edge=right.tolist())
        self._measure_cache[key] = dict(result)
        return result

    def route(self, chord, width):
        key = (chord.wkb, float(width))
        if key in self._route_cache:
            return self._route_cache[key]
        owner = self
        vector = np.subtract(chord.coords[-1], chord.coords[0])
        vector /= max(np.linalg.norm(vector), 1e-9)
        normal = np.array([-vector[1], vector[0]])
        class DirectionalCost:
            def values(self, xy):
                p, m, s = owner.values(xy)
                # A fixed-width directional edge probe uses cached width, never estimates a new width.
                edges = []
                for sign in (-1, 1):
                    at = xy + sign*normal*width/2
                    gx, gy = owner.image.sample(owner.image.gx, at), owner.image.sample(owner.image.gy, at)
                    edges.append(np.clip(np.abs(gx*normal[0]+gy*normal[1])/owner.image.scale, 0, 1))
                rgb = np.nan_to_num(np.minimum(*edges))
                model = np.fmax(p, m)
                score = .7*np.nan_to_num(model) + .2*rgb + .1*s
                return np.where(np.isfinite(model), score, 0.)
        evidence = ConnectionEvidence(probability=DirectionalCost())
        route = _evidence_route(chord, max(4., min(width, 18.)), evidence)
        if route is None:
            self._route_cache[key] = None
            return None
        # Remove grid stair steps; keep endpoints pinned. Validate the actual simplified curve later.
        result = LineString(route).simplify(self.image.resolution, preserve_topology=True)
        self._route_cache[key] = result
        return result

    def ribbons(self, first, second):
        _, xy, _, _ = stations(first, self.config["sample_m"])
        other = np.array([second.interpolate(second.project(p)).coords[0] for p in points(xy)])
        delta = other - xy
        distances = np.linalg.norm(delta, axis=1)
        normal = delta / np.maximum(distances[:, None], 1e-9)
        fractions = np.linspace(-.5, 1.5, 61)
        probes = xy[:, None, :] + delta[:, None, :] * fractions[None, :, None]
        p, m, s = self.values(probes.reshape(-1, 2))
        model = np.fmax(p, m).reshape(len(xy), -1)
        middle = (fractions >= .15) & (fractions <= .85)
        a, b = np.argmin(abs(fractions)), np.argmin(abs(fractions-1))
        covered = np.isfinite(model).all(axis=1) & (distances >= 1.)
        both = (model[:, a] >= .2) & (model[:, b] >= .2)
        ribbon = (model[:, middle] >= .2).mean(axis=1) >= .9
        # Stable non-road strip at least 1.5m wide between supported axes.
        holes = np.array([longest_hole(row >= .1, distances[i]*.7) for i, row in enumerate(model[:, middle])])
        separate = both & (holes >= 1.5)
        gray = self.image.sample(self.image.gray, probes.reshape(-1, 2)).reshape(model.shape)
        # A stable intensity strip between two similar road interiors vetoes merging
        # even when a blurred probability map fills the divider. This is a conservative
        # separator cue, not a semantic classifier for vegetation/shadows.
        middle_gray = np.median(gray[:, middle], axis=1)
        contrast = np.minimum(abs(middle_gray-gray[:, a]), abs(middle_gray-gray[:, b]))
        raw_separator = (contrast >= .25) & (abs(gray[:, a]-gray[:, b]) < .2) & (distances >= 3.)
        return dict(valid_fraction=float(covered.mean()),
                    single_ribbon_fraction=float((covered & both & ribbon & ~raw_separator).mean()),
                    two_ribbon_fraction=float((covered & separate).mean()),
                    raw_separator_fraction=float((covered & both & raw_separator).mean()),
                    rgb_valid_fraction=float(np.isfinite(gray).all(axis=1).mean()),
                    separation_m=float(np.median(distances)),
                    cross_sections=dict(fractions=fractions.tolist(), probability=p.reshape(model.shape).tolist(),
                        molra=m.reshape(model.shape).tolist(), formal_surface=s.reshape(model.shape).tolist(),
                        gray=gray.tolist(),
                        stations_xy=xy.tolist(), separation_m=distances.tolist()))

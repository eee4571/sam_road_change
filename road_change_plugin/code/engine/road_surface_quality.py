"""Conservative final-surface regularization from saved observations, in metres.

No new image analysis: width grades, probability and other period axes are read
only. Unknown evidence is never treated as negative evidence for road removal.
"""
import numpy as np
from scipy.ndimage import median_filter
from shapely import line_interpolate_point
from shapely.geometry import LineString, Polygon
from shapely.strtree import STRtree


def surface_axis(axis, width, evidence=None):
    """Bounded cartographic axis, used only to remove sub-metre edge chatter.

    This is not fold repair or a new road decision. Endpoints are pinned; the
    caller verifies contacts. Positive existing surface evidence is required
    for smoothing, and an unsupported original station may not become a reason
    to move a supported station outside the observed road.
    """
    from scipy.ndimage import gaussian_filter1d
    from shapely import covers, points
    simplified=axis.simplify(.2,preserve_topology=True)
    ev=evidence() if callable(evidence) else evidence
    if ev is None or ev.surface is None or axis.length<12:
        return simplified
    s=np.linspace(0.,axis.length,max(3,int(np.ceil(axis.length))+1))
    xy=np.asarray([p.coords[0] for p in line_interpolate_point(axis,s)])
    smooth=gaussian_filter1d(xy,max(2.,min(5.,width*.4)),axis=0,mode='nearest')
    delta=smooth-xy
    norm=np.linalg.norm(delta,axis=1)
    delta*=np.minimum(1.,min(.75,width*.08)/np.maximum(norm,1e-9))[:,None]
    taper=np.sin(np.clip(np.minimum(s,axis.length-s)/max(6.,width),0,1)*np.pi/2)**2
    candidate=xy+delta*taper[:,None]
    candidate[0]=xy[0];candidate[-1]=xy[-1]
    before=covers(ev.surface.context,points(xy))
    after=covers(ev.surface.context,points(candidate))
    if np.any(before & ~after):return simplified
    result=LineString(candidate).simplify(.08,preserve_topology=True)
    return result if result.is_simple and result.hausdorff_distance(axis)<=.8 else simplified


def reliable_width(row):
    grade = str(row.get('quality_grade', row.get('width_quality', ''))).upper()
    if not grade or grade == 'NAN':
        return None
    width = float(row.get('width_m', 0))
    ratio = float(row.get('valid_ratio', 1))
    scatter = float(row.get('width_std', 0))
    return grade in ('A', 'B') and ratio >= .15 and scatter <= .6 * width


def stable_width(stations, values, reliable):
    """One robust width per junction chain, split only by sustained evidence.

    Uniform sampling makes the representative independent of feature splits.
    Low-quality holes interpolate from reliable samples before segmentation;
    they cannot create a new narrow section or terminate the road surface.
    """
    s = np.asarray(stations, float)
    v = np.asarray(values, float)
    good = np.asarray(reliable, bool) & np.isfinite(v) & (v > 0)
    if not good.any():
        valid = np.isfinite(v) & (v > 0)
        if not valid.any():
            raise ValueError('Road surface has no positive representative width')
        result = np.full(len(v), np.median(v[valid]))
        return result, dict(method='unverified_representative', reliable_fraction=0., sections=1)
    filled = np.interp(s, s[good], v[good])
    representative = float(np.median(filled))
    step = max(float(np.median(np.diff(s))), .1)
    # A short wobble cannot become a width zone. Use physical length, not the
    # number or order of source features, as the persistence criterion.
    minimum = max(24., 3. * representative)
    filtered = median_filter(filled, size=max(3, int(8. / step) | 1), mode='nearest')
    threshold = max(1.5, .20 * representative)
    window=max(2,int(np.ceil(minimum/step)))
    candidates=[]
    for k in range(window,len(s)-window):
        left,right=filtered[k-window:k],filtered[k:k+window]
        lm,rm=float(np.median(left)),float(np.median(right))
        if abs(lm-rm)<threshold:continue
        # Both plateaux need actual A/B observations, not interpolated holes.
        if good[k-window:k].mean()<.6 or good[k:k+window].mean()<.6:continue
        if np.quantile(abs(left-lm),.8)>threshold/2 or np.quantile(abs(right-rm),.8)>threshold/2:continue
        loss=float(np.mean(abs(left-lm))+np.mean(abs(right-rm)))
        candidates.append((loss,k))
    boundaries=[]
    for _,k in sorted(candidates):
        if all(abs(s[k]-s[j])>=minimum for j in boundaries):boundaries.append(k)
    edges=[0,*sorted(boundaries),len(s)]
    # Collapse a candidate if the resulting full sections do not actually have
    # different representative widths. Local noise cannot create an extra zone.
    changed=True
    while changed:
        changed=False
        for i in range(1,len(edges)-1):
            if abs(np.median(filled[edges[i-1]:edges[i]])-np.median(filled[edges[i]:edges[i+1]]))<threshold:
                edges.pop(i);changed=True;break
    zones=list(zip(edges[:-1],edges[1:]))
    out = np.empty(len(v))
    levels = []
    for a, b in zones:
        level = float(np.median(filled[a:b]))
        out[a:b] = level
        levels.append(level)
    # Monotone cubic transitions, zero slope at both ends; no width overshoot.
    for i in range(len(zones) - 1):
        k = zones[i][1]
        span = min(max(6., representative), (s[k] - s[zones[i][0]]) / 3,
                   (s[zones[i + 1][1] - 1] - s[k]) / 3)
        if span <= 0:
            continue
        selected = abs(s - s[k]) <= span
        t = np.clip((s[selected] - s[k] + span) / (2 * span), 0, 1)
        out[selected] = levels[i] + (levels[i + 1] - levels[i]) * t * t * (3 - 2 * t)
    return out, dict(method='junction_chain_representative', sections=len(zones),
                     reliable_fraction=float(good.mean()), filled_samples=int((~good).sum()),
                     width_before_range=[float(v.min()), float(v.max())],
                     width_after_range=[float(out.min()), float(out.max())])


def junction_footprint(arms):
    """Tangent portal curves at a real junction; never close an entire network.

    Each arm is (outward axis, endpoint width). Flat portals lie inside the
    existing branch corridor. Quadratic corners connect their side tangents.
    """
    portals = []
    for axis, width in arms:
        reach = min(axis.length * .45, max(2., width))
        p = np.asarray(axis.interpolate(reach).coords[0])
        inside = np.asarray(axis.interpolate(max(0., reach - 1.)).coords[0])
        d = p - inside
        d /= max(np.linalg.norm(d), 1e-9)
        n = np.array([-d[1], d[0]]) * width / 2
        portals.append((np.arctan2(d[1], d[0]), p - n, p + n, d, reach))
    portals.sort(key=lambda p: p[0])
    boundary = []
    for i, (_, right, left, d, reach) in enumerate(portals):
        boundary.extend([right, left])
        _, q, _, e, next_reach = portals[(i + 1) % len(portals)]
        matrix = np.column_stack([d, -e])
        if abs(np.linalg.det(matrix)) < .05:
            continue
        t, u = np.linalg.solve(matrix, left - q)
        if not (0 <= t <= 2 * reach and 0 <= u <= 2 * next_reach):
            continue
        control = left - t * d
        f = np.linspace(0., 1., 9)[1:-1, None]
        boundary.extend((1-f)**2 * left + 2*f*(1-f)*control + f**2*q)
    polygon = Polygon(boundary)
    return polygon if polygon.is_valid else Polygon()


def short_noise_indices(axes, widths, measurements, evidence, other_periods=(), protected=()):
    """Drop only an isolated *component* with jointly negative saved evidence.

    No temporal observations / missing probability / missing RGB grades means
    retain. Event-constrained roads and roads near another component are pinned.
    """
    tree = STRtree(axes)
    parent = list(range(len(axes)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, axis in enumerate(axes):
        for j in tree.query(axis, predicate='dwithin', distance=.5):
            parent[root(int(j))] = root(i)
    groups = {}
    for i in range(len(axes)):
        groups.setdefault(root(i), []).append(i)
    temporal = [STRtree(lines) for lines in other_periods]
    protected = set(protected)
    dropped, audit = set(), []
    for ids in groups.values():
        length = sum(axes[i].length for i in ids)
        width = float(np.median([widths[i] for i in ids]))
        if length > min(30., 3 * width):
            continue
        record = dict(features=ids, length_m=length, action='retain', reason='insufficient_evidence')
        nearby = {int(j) for i in ids for j in tree.query(axes[i], predicate='dwithin', distance=max(6., width))} - set(ids)
        if protected.intersection(ids):
            record['reason'] = 'event_constrained'
        elif nearby:
            record['reason'] = 'near_network'
        elif temporal and any(len(t.query(a, predicate='dwithin', distance=max(2., width / 2))) for t in temporal for a in [axes[i] for i in ids]):
            record['reason'] = 'other_period_support'
        elif temporal:
            rgb = [measurements.profile_quality(axes[i], widths[i])[2] for i in ids]
            # Only explicitly C/invalid observations count against the road.
            rejected_rgb = all(np.all(q == 0) for q in rgb)
            if rejected_rgb:
                ev = evidence() if callable(evidence) else evidence
                scores = [ev.measure(axes[i]) for i in ids] if ev is not None else []
                record['evidence']=scores
                # The published surface may itself be a width corridor. Do not
                # use that derived polygon as independent positive RGB evidence.
                if scores and all(s['probability_valid_fraction'] >= .9 and s['probability_support'] < .1
                                  for s in scores):
                    dropped.update(ids)
                    record.update(action='suppress', reason='isolated_short_jointly_unsupported')
        audit.append(record)
    return dropped, audit

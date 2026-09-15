"""Second-stage spatial reconstruction of saved observations; no image inference."""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

from .raw_boundary_core import digest, save_json


@dataclass
class ReconstructionConfig:
    transient_m: float = 30.0
    flank_m: float = 15.0
    excursion_m: float = 2.5
    short_gap_m: float = 18.0
    medium_gap_m: float = 75.0
    observation_huber_m: float = 1.0
    first_weight: float = .35
    second_weight: float = 2.5
    junction_weight: float = .25
    junction_observation_weight: float = .20
    max_iterations: int = 350


def spans(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def detect_outliers(s, y, confidence, usable, junction, config):
    """Find bounded excursions using two stable flanks, including one-sided plateaus.

    Duration is in metres. A lasting step or monotonic ramp has no returning flank.
    Near junctions require a larger excursion; genuine short bays remain ambiguous.
    """
    n = len(s)
    reasons = np.full(n, '', dtype=object)
    for start in range(n):
        before = np.flatnonzero(usable & (s < s[start]) & (s >= s[start]-config.flank_m))
        if len(before) < 3:
            continue
        a = np.median(y[before])
        mad_a = np.median(np.abs(y[before]-a))
        if mad_a > .9:
            continue
        for end in range(start+1, n):
            if s[end]-s[start] > config.transient_m:
                break
            middle = np.arange(start, end)
            middle = middle[usable[middle]]
            if not len(middle):
                continue
            after = np.flatnonzero(usable & (s >= s[end]) & (s < s[end]+config.flank_m))
            if len(after) < 3:
                continue
            b = np.median(y[after])
            mad_b = np.median(np.abs(y[after]-b))
            if mad_b > .9 or abs(a-b) > 1.5:
                continue
            threshold = max(config.excursion_m, 3*1.4826*max(mad_a, mad_b))
            if junction[start:end].any():
                threshold *= 2
            target = (a+b)/2
            difference = y[middle]-target
            # Both signs supported; require most observations to share the excursion.
            direction = np.sign(np.median(difference))
            abnormal = difference*direction > threshold
            if abnormal.mean() >= .8:
                reasons[middle[abnormal]] = 'short_returning_excursion'
    # Low-evidence measurements conflicting with a supported local trend.
    clean = usable & (reasons == '') & (confidence >= .16)
    for i in np.flatnonzero(usable & (confidence < .20) & (reasons == '')):
        near = np.flatnonzero(clean & (np.abs(s-s[i]) <= 24) & (np.arange(n) != i))
        if len(near) < 5 or not (np.any(s[near] < s[i]) and np.any(s[near] > s[i])):
            continue
        slope, intercept = np.polyfit(s[near]-s[i], y[near], 1)
        residual = y[near]-(slope*(s[near]-s[i])+intercept)
        noise = np.median(np.abs(residual-np.median(residual)))
        threshold = max(3., 4*1.4826*noise)*(2 if junction[i] else 1)
        if abs(y[i]-intercept) > threshold:
            reasons[i] = 'low_confidence_trend_conflict'
    return reasons


def huber_value_gradient(residual, delta):
    absolute = np.abs(residual)
    return (np.where(absolute <= delta, .5*residual**2, delta*(absolute-.5*delta)),
            np.clip(residual, -delta, delta))


def smooth_side(s, y, confidence, anchor, junction, config):
    """Bounded Huber data fit + Huber first/second spatial differences.

    Gap priors have explicit provenance, use only this chain's anchors, and are weak
    compared with observations. Long gaps hold endpoint neighbourhood medians and
    blend across the middle; no unsupported linear extrapolation is performed.
    """
    n = len(s)
    indices = np.flatnonzero(anchor)
    if not len(indices):
        return np.full(n, np.nan), np.full(n, 'unresolved', object), np.zeros(n), True
    source = np.full(n, 'smoothed', object)
    nearest = np.abs(s[:, None]-s[indices][None, :]).argmin(axis=1)
    distance = np.abs(s-s[indices][nearest])
    certainty = confidence[indices][nearest].copy()
    seed = np.interp(s, s[indices], y[indices])
    pchip = PchipInterpolator(s[indices], y[indices], extrapolate=False) if len(indices) >= 2 else None
    for begin, end in spans(~anchor):
        inside = begin > 0 and end < n
        gap = s[end]-s[begin-1] if inside else np.inf
        block = np.arange(begin, end)
        if inside and gap <= config.medium_gap_m:
            if gap <= config.short_gap_m:
                u = (s[block]-s[begin-1])/gap
                seed[block] = y[begin-1]+(y[end]-y[begin-1])*(u*u*(3-2*u))
            else:
                seed[block] = pchip(s[block])
            source[block] = 'interpolated'
            certainty[block] = min(confidence[begin-1], confidence[end])*.7*np.exp(-distance[block]/90)
        else:
            source[block] = 'propagated'
            if inside:
                l = indices[(s[indices] <= s[begin-1]) & (s[indices] >= s[begin-1]-18)]
                r = indices[(s[indices] >= s[end]) & (s[indices] <= s[end]+18)]
                a, b = np.median(y[l]), np.median(y[r])
                u = np.clip(((s[block]-s[begin-1])/gap-.25)*2, 0, 1)
                seed[block] = a+(b-a)*u*u*(3-2*u)
            else:
                edge = indices[0] if begin == 0 else indices[-1]
                near = indices[np.abs(s[indices]-s[edge]) <= 18]
                seed[block] = np.median(y[near])
            certainty[block] = np.minimum(.20, certainty[block]*.45)*np.exp(-distance[block]/120)
    certainty[junction] *= .65
    certainty *= min(1., len(indices)/3)
    if n == 1:
        return seed, source, certainty, True
    scale = 3/np.maximum(np.diff(s), .01)
    edge_weight = np.where(junction[:-1] | junction[1:], config.junction_weight, 1.)
    curvature_weight = np.minimum(edge_weight[:-1], edge_weight[1:])
    weights = np.where(anchor, np.clip((confidence/.35)**2, .3, 8.), 0.)
    weights[junction] *= config.junction_observation_weight
    obs = np.nan_to_num(y)
    prior_weight = np.where(anchor, 0., .12)

    def objective(x):
        val, grad = huber_value_gradient(x-obs, config.observation_huber_m)
        loss = np.dot(weights, val)+.5*np.dot(prior_weight, (x-seed)**2)
        gradient = weights*grad+prior_weight*(x-seed)
        d1 = np.diff(x)*scale
        val, grad = huber_value_gradient(d1, .6)
        loss += config.first_weight*np.dot(edge_weight, val)
        flow = config.first_weight*edge_weight*grad*scale
        gradient[:-1] -= flow
        gradient[1:] += flow
        if n > 2:
            val, grad = huber_value_gradient(np.diff(d1), .4)
            loss += config.second_weight*np.dot(curvature_weight, val)
            flow2 = config.second_weight*curvature_weight*grad
            first_gradient = np.zeros(n-1)
            first_gradient[:-1] -= flow2
            first_gradient[1:] += flow2
            first_gradient *= scale
            gradient[:-1] -= first_gradient
            gradient[1:] += first_gradient
        return loss, gradient

    result = minimize(objective, np.clip(seed, .75, 20), jac=True, method='L-BFGS-B',
                      bounds=[(.75, 20.)]*n,
                      options={'maxiter': config.max_iterations, 'ftol': 1e-9, 'gtol': 1e-5})
    certainty[anchor] *= np.exp(-np.abs(result.x[anchor]-y[anchor])/4)
    return result.x, source, certainty, bool(result.success)


def reconstruct_chain(group, config):
    group = group.sort_values('s_m').copy()
    s = group.s_m.to_numpy()
    if len(s) > 1 and np.any(np.diff(s) <= 0):
        raise ValueError('Each chain must have strictly increasing distances')
    flags = group['flags'].fillna('').astype(str)
    junction = flags.str.contains('junction').to_numpy()
    hard = flags.str.contains('image_or_nodata_boundary|no_valid_pair|search_limit|crossing').to_numpy()
    accepted = group.accepted.to_numpy(bool)
    side_sources, side_conf, reason_parts, converged = [], [], [], []
    for side in ('left', 'right'):
        y = group[f'optimized_{side}_distance'].to_numpy(float)
        confidence = group[f'optimized_{side}_confidence'].fillna(0).to_numpy(float)
        usable = np.isfinite(y) & ~hard & (confidence >= .08)
        reasons = detect_outliers(s, y, confidence, usable, junction, config)
        # Junction observations remain available, but receive weaker smoothing and confidence.
        anchor = usable & (reasons == '') & (accepted | (junction & (confidence >= .16)))
        final, source, certainty, success = smooth_side(s, y, confidence, anchor, junction, config)
        source[anchor & accepted & (np.abs(final-y) <= .25)] = 'measured'
        group[f'final_{side}_distance'] = final
        group[f'final_{side}_confidence'] = certainty
        group[f'{side}_source'] = source
        group[f'{side}_anchor'] = anchor
        group[f'{side}_outlier_reason'] = reasons
        side_sources.append(source)
        side_conf.append(certainty)
        reason_parts.append(np.where(reasons == '', '', side+':'+reasons))
        converged.append(success)
    group['final_width'] = group.final_left_distance+group.final_right_distance
    rank = {'measured': 0, 'smoothed': 1, 'interpolated': 2, 'propagated': 3, 'unresolved': 4}
    group['width_source'] = [max(pair, key=rank.get) for pair in zip(*side_sources)]
    group['final_confidence'] = np.minimum(*side_conf)
    group['outlier_reason'] = [';'.join(r for r in pair if r) for pair in zip(*reason_parts)]
    group['reconstruction_available'] = np.isfinite(group.final_width)
    group['solver_converged'] = all(converged)
    group['reconstruction_flags'] = np.where(group.reconstruction_available, '', 'no_reliable_chain_support')
    for side, sign in [('left', 1), ('right', -1)]:
        for axis in ('x', 'y'):
            group[f'final_{side}_{axis}'] = group[f'center_{axis}']+sign*group[f'normal_{axis}']*group[f'final_{side}_distance']
    return group


def jump_stats(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return dict(pairs=len(v), mean_m=float(v.mean()) if len(v) else None,
                p95_m=float(np.percentile(v, 95)) if len(v) else None)


def summarize(frame):
    raw_jumps, final_same, final_all, raw_diagnostic_same, final_diagnostic_same = [], [], [], [], []
    raw_length = final_length = total_length = 0.
    outlier_runs = 0
    for _, g in frame.groupby('road_id', sort=False):
        ds = np.diff(g.s_m)
        a = g.accepted.to_numpy(bool)
        f = g.reconstruction_available.to_numpy(bool)
        matched = a[:-1] & a[1:] & f[:-1] & f[1:]
        raw_jumps.extend(np.abs(np.diff(g.optimized_width))[matched])
        final_same.extend(np.abs(np.diff(g.final_width))[matched])
        final_all.extend(np.abs(np.diff(g.final_width))[f[:-1] & f[1:]])
        raw_finite = np.isfinite(g.optimized_width.to_numpy())
        diagnostic_pairs = raw_finite[:-1] & raw_finite[1:] & f[:-1] & f[1:]
        raw_diagnostic_same.extend(np.abs(np.diff(g.optimized_width))[diagnostic_pairs])
        final_diagnostic_same.extend(np.abs(np.diff(g.final_width))[diagnostic_pairs])
        total_length += ds.sum()
        raw_length += ds[a[:-1] & a[1:]].sum()
        final_length += ds[f[:-1] & f[1:]].sum()
        outlier_runs += len(spans(g.outlier_reason.ne('').to_numpy()))
    counts = frame.width_source.value_counts().to_dict()
    return dict(samples=len(frame), roads=int(frame.road_id.nunique()),
        original_valid_sample_fraction=float(frame.accepted.mean()),
        continuous_sample_fraction=float(frame.reconstruction_available.mean()),
        final_confidence_ge_016_fraction=float((frame.reconstruction_available & (frame.final_confidence >= .16)).mean()),
        sampled_chain_length_m=total_length,
        original_valid_length_fraction=raw_length/total_length,
        continuous_length_fraction=final_length/total_length,
        outlier_points=int(frame.outlier_reason.ne('').sum()), outlier_segments=outlier_runs,
        source_counts={str(k): int(v) for k, v in counts.items()},
        source_fractions={str(k): float(v/len(frame)) for k, v in counts.items()},
        original_adjacent_jumps=jump_stats(raw_jumps), final_same_pairs_jumps=jump_stats(final_same),
        final_all_adjacent_jumps=jump_stats(final_all),
        raw_diagnostic_same_pairs_jumps=jump_stats(raw_diagnostic_same),
        final_diagnostic_same_pairs_jumps=jump_stats(final_diagnostic_same),
        unconverged_roads=int(frame.loc[~frame.solver_converged, 'road_id'].nunique()),
        note='Coverage is reconstruction availability, not verified boundary accuracy. Sources are conservative across both sides.')


def main(input_dir, output_dir, config):
    output_dir.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(input_dir/'samples.csv')
    groups = []
    for i, (_, group) in enumerate(frame.groupby('road_id', sort=False)):
        groups.append(reconstruct_chain(group, config))
        if (i+1) % 500 == 0:
            print(f'Reconstructed {i+1} chains', flush=True)
    result = pd.concat(groups, ignore_index=True)
    result.to_csv(output_dir/'samples.csv', index=False)
    result.to_json(output_dir/'samples.json', orient='records', indent=2, force_ascii=False)
    manifest = json.loads((input_dir/'manifest.json').read_text(encoding='utf-8'))
    manifest.update(reconstruction_config=asdict(config), observations=str((input_dir/'samples.csv').resolve()),
                    observations_sha256=digest(input_dir/'samples.csv'))
    save_json(output_dir/'manifest.json', manifest)
    summary = summarize(result)
    save_json(output_dir/'summary.json', summary)
    columns = ['road_id', 'sample_id', 's_m', 'accepted', 'flags', 'optimized_left_distance',
               'optimized_right_distance', 'optimized_width', 'optimized_confidence',
               'final_left_distance', 'final_right_distance', 'final_width', 'final_confidence',
               'width_source', 'outlier_reason']
    result[columns].to_csv(output_dir/'width_profiles.csv', index=False)
    segment_rows, road_rows = [], []
    for road_id, g in result.groupby('road_id', sort=False):
        road_rows.append(dict(road_id=road_id, samples=len(g), original_coverage=g.accepted.mean(),
                              final_coverage=g.reconstruction_available.mean(),
                              outliers=g.outlier_reason.ne('').sum(), mean_confidence=g.final_confidence.mean()))
        for begin, end in spans(g.outlier_reason.ne('').to_numpy()):
            part = g.iloc[begin:end]
            segment_rows.append(dict(road_id=road_id, start_m=part.s_m.iloc[0], end_m=part.s_m.iloc[-1],
                                     points=len(part), reasons=';'.join(sorted(set(part.outlier_reason)))))
    pd.DataFrame(road_rows).to_csv(output_dir/'road_summary.csv', index=False)
    pd.DataFrame(segment_rows, columns=['road_id', 'start_m', 'end_m', 'points', 'reasons']).to_csv(output_dir/'outlier_segments.csv', index=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, ReconstructionConfig())

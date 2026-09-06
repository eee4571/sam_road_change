"""Symmetric, recall-first longitudinal candidates on immutable final roads.

Geometry defines intervals; station evidence annotates them and excludes known
present/present spans. Weak evidence, invalid observations and junctions are QA
states, not deletion rules. This module has no reference-truth input.
"""
from collections import Counter
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from shapely import from_wkt, make_valid, union_all
from shapely.geometry import LineString
from shapely.ops import substring
from shapely.strtree import STRtree


def line_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    return [p for child in getattr(geometry, "geoms", ()) for p in line_parts(child)]


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end-start <= 1e-7:
            continue
        if merged and start <= merged[-1][1]+1e-7:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


class LongitudinalCoverage:
    """Exact line/buffer intersections, restricted to locally aligned road edges."""

    def __init__(self, lines, tolerance):
        self.tolerance = tolerance
        self.edges, self.directions = [], []
        for line in lines:
            coords = np.asarray(line.coords)[:, :2]
            for a, b in zip(coords[:-1], coords[1:]):
                direction = b-a
                length = np.linalg.norm(direction)
                if length > 1e-7:
                    self.edges.append(LineString([a, b]))
                    self.directions.append(direction/length)
        self.tree = STRtree(self.edges)
        self.directions = np.asarray(self.directions)

    def uncovered(self, axis, tolerance=None):
        tolerance = self.tolerance if tolerance is None else tolerance
        covered, offset = [], 0.
        coords = np.asarray(axis.coords)[:, :2]
        for a, b in zip(coords[:-1], coords[1:]):
            edge = LineString([a, b])
            length = edge.length
            if length <= 1e-7:
                continue
            ids = self.tree.query(edge, predicate="dwithin", distance=tolerance)
            if len(ids):
                aligned = ids[np.abs(self.directions[ids] @ ((b-a)/length)) >= .90]
                if len(aligned):
                    support = union_all([self.edges[i].buffer(tolerance) for i in aligned])
                    for part in line_parts(edge.intersection(support)):
                        positions = [float(np.dot(np.asarray(p)[:2]-a, (b-a)/length)) for p in part.coords]
                        covered.append((offset+max(0., min(positions)), offset+min(length, max(positions))))
            offset += length
        result, position = [], 0.
        for start, end in merge_intervals(covered):
            if start > position+1e-7:
                result.append((position, start))
            position = max(position, end)
        if position < axis.length-1e-7:
            result.append((position, axis.length))
        return result


def presence_seeds(axis, rows, source, coverage, kind, minimum_length, minimum_area):
    """Emit entire admissible intervals irrespective of confidence transitions."""
    intervals = coverage.uncovered(axis)
    side = "after" if kind == "added" else "before"
    other = "before" if kind == "added" else "after"
    spacing = axis.length/len(rows)
    records, audit = [], []
    counts = Counter({f"{kind}_source_axes": 1, f"{kind}_uncovered_intervals": len(intervals)})
    counts[f"{kind}_uncovered_length_m"] = sum(b-a for a, b in intervals)
    # No confidence label is used to split adjacent candidate cells. Only known
    # opposite-period presence is a barrier; in particular junctions are not.
    admissible = merge_intervals([(i*spacing, (i+1)*spacing) for i, r in enumerate(rows)
                                 if r[f"{other}_state"] != "present" and not r[f"{other}_geometry"]])
    for interval_id, (start, end) in enumerate(intervals):
        accepted_spans = [(max(start, a), min(end, b)) for a, b in admissible
                          if min(end, b)-max(start, a) > 1e-7]
        accepted_length = sum(b-a for a, b in accepted_spans)
        counts[f"{kind}_present_present_excluded_length_m"] += end-start-accepted_length
        audit.append(dict(candidate_type=kind, axis_id=rows[0]["axis_id"], interval_id=interval_id,
                          stage="geometry_uncovered", start_m=start, end_m=end, length_m=end-start,
                          accepted=bool(accepted_spans), retained_length_m=accepted_length,
                          audit_reason="uncovered_final_road" if accepted_spans else "opposite_period_present",
                          geometry=substring(axis, start, end)))
        for a, b in accepted_spans:
            local_axis = substring(axis, a, b)
            selected = [r for i, r in enumerate(rows) if min(b, (i+1)*spacing)-max(a, i*spacing) > 1e-7]
            lengths = np.array([min(b, (i+1)*spacing)-max(a, i*spacing) for i, r in enumerate(rows)
                                if min(b, (i+1)*spacing)-max(a, i*spacing) > 1e-7])
            width = float(np.median([source.width(axis.interpolate(r["station_m"])) for r in selected]))
            geometry = local_axis.buffer(width/2, cap_style="flat")
            reasons = {r[f"{other}_reason"] for r in selected}
            known_source = all(r[f"{side}_state"] == "present" for r in selected)
            known_absent = all(r[f"{other}_state"] == "absent" for r in selected)
            qa = "confirmed" if known_source and known_absent else "probable" if known_source else "uncertain"
            junction = any(r["junction"] for r in selected)
            if junction:
                reasons.add("junction_evidence_discount")
                if qa == "confirmed":
                    qa = "probable"
            if b-a < minimum_length:
                reasons.add("short_interval_retained")
                qa = "uncertain"
            if geometry.area < minimum_area:
                reasons.add("small_area_retained")
                qa = "uncertain"
            if not known_source:
                reasons.add("source_observation_uncertain")
            reasons.add("longitudinal_unmatched_final_road")
            confidence = {"confirmed": .90, "probable": .60, "uncertain": .30}[qa] - (.05 if junction else 0.)
            evidence = {}
            for period in ("before", "after"):
                for field in ("geometry", "surface", "probability", "valid"):
                    evidence[f"{period}_{field}_ratio"] = float(np.average([r[f"{period}_{field}"] for r in selected], weights=lengths))
            record = dict(change_typ=kind, width_bef=width if kind == "removed" else 0.,
                          width_aft=width if kind == "added" else 0., width_diff=0.,
                          length_m=b-a, axis_wkt=local_axis.wkt, source_axis=rows[0]["axis_id"],
                          start_m=a, end_m=b, confidence=confidence, qa_state=qa,
                          audit_reason=";".join(sorted(reasons)), junction=junction,
                          **evidence, geometry=geometry)
            records.append(record)
            counts[f"{kind}_{qa}_seeds"] += 1
            for r in selected:
                r["existence_pass"] = True
                r["continuity_pass"] = b-a >= minimum_length
            audit.append(dict(candidate_type=kind, axis_id=rows[0]["axis_id"], interval_id=interval_id,
                              stage="local_seed", start_m=a, end_m=b, length_m=b-a, accepted=True,
                              confidence=confidence, qa_state=qa, audit_reason=record["audit_reason"],
                              **evidence, geometry=local_axis))
    return records, audit, counts


@dataclass(frozen=True)
class PresenceAcceptance:
    minimum_valid_ratio: float = .95
    minimum_source_support: float = .60
    minimum_absent_ratio: float = .90
    opposing_corridor_ratio: float = .50
    opposing_corridor_distance_m: float = 30.


def qualify_presence_candidates(candidates, scenes, evidence, *, minimum_length=24., minimum_area=4.,
                                config=PresenceAcceptance()):
    """Keep recall candidates auditable, publish only supported, unambiguous ones.

    Source geometry alone is not independent road evidence. Near-parallel added
    and removed tracks are sent to review, never merged into a guessed track.
    Width candidates bypass this presence-only qualification.
    """
    result = candidates.copy().reset_index(drop=True)
    result["candidate_id"] = np.arange(len(result))
    result["candidate_qa_state"] = result.qa_state
    result["publication_state"] = "accepted"
    result["precision_reason"] = "paired_width_unchanged"
    from .auto_track_evidence import displacement_rescue, longest_run, nearby_opposite_support
    grouped = {(side, int(axis_id)): rows.sort_values("station_m")
               for (side, axis_id), rows in evidence.groupby(["side", "axis_id"])} if len(evidence) else {}
    axes = {}
    for index, row in result.loc[result.change_typ.isin(["added", "removed"])].iterrows():
        side = "after" if row.change_typ == "added" else "before"
        other = "before" if side == "after" else "after"
        source = scenes[side]
        axis = from_wkt(row.axis_wkt)
        axes[index] = axis
        rows = grouped.get((side, int(row.source_axis)))
        reasons = []
        if rows is None or rows.empty:
            reasons.append("missing_source_evidence")
            valid = support = absent = absence_run = nonjunction_run = junction_ratio = 0.
        else:
            spacing = source.lines[int(row.source_axis)].length/len(rows)
            weight = np.maximum(0., np.minimum(float(row.end_m), rows.station_m.to_numpy()+spacing/2)
                                - np.maximum(float(row.start_m), rows.station_m.to_numpy()-spacing/2))
            def mean(values):
                return float(np.average(np.asarray(values, dtype=float), weights=weight)) if weight.sum() else 0.
            valid = mean(rows[f"{side}_valid"].astype(bool) & rows[f"{other}_valid"].astype(bool))
            # Require corroboration from the finalized surface and the source
            # probability distribution, not just its pre-existing centerline.
            supported = ((rows[f"{side}_surface"] >= .55)
                         & ((rows[f"{side}_scene_percentile_rank"] >= .75) | rows[f"{side}_probability"].astype(bool)))
            support = mean(supported)
            absent = mean(rows[f"{other}_state"] == "absent")
            absence_run = longest_run(rows[f"{other}_state"].eq('absent').to_numpy(),weight)
            nonjunction_run = longest_run(~rows.junction.to_numpy(bool),weight)
            junction_ratio = mean(rows.junction.to_numpy(bool))
        result.loc[index, "source_support_ratio"] = support
        result.loc[index, "opposite_absent_ratio"] = absent
        result.loc[index, "paired_valid_ratio"] = valid
        result.loc[index, "opposite_absence_run_m"] = absence_run
        result.loc[index, "nonjunction_run_m"] = nonjunction_run
        if valid < config.minimum_valid_ratio:
            reasons.append("incomplete_paired_observation")
        if support < config.minimum_source_support:
            reasons.append("source_road_not_corroborated")
        if absent < config.minimum_absent_ratio or absence_run < minimum_length:
            reasons.append("opposite_absence_not_sustained")
        if junction_ratio > 0. and nonjunction_run < minimum_length:
            reasons.append("junction_only_presence_change")
        displacement = displacement_rescue(axis,source,scenes[other])
        for key,value in displacement.items():
            result.loc[index,key] = value
        if displacement['displacement_rescue']:
            reasons.append('same_road_displacement_rescue')
        elif displacement['displacement_track_ambiguous']:
            reasons.append('cross_track_presence_ambiguity')
        if row.length_m < minimum_length or row.geometry.area < minimum_area:
            reasons.append("short_or_small_candidate")
        # The original uncut candidate remains in the review layer. Publication
        # follows the actual final surface and observation footprint.
        clipped = make_valid(row.geometry).intersection(make_valid(source.surface(axis)))
        clipped = clipped.intersection(source.valid).intersection(scenes[other].valid)
        if clipped.area < minimum_area:
            reasons.append("no_publishable_source_surface")
        if not reasons:
            nearby=nearby_opposite_support(axis,scenes[other],float(row.width_aft if side=='after' else row.width_bef))
            for key,value in nearby.items(): result.loc[index,key]=value
            if nearby['nearby_opposite_reason']: reasons.append(nearby['nearby_opposite_reason'])
        if reasons:
            result.loc[index, "publication_state"] = "review"
        else:
            result.at[index, "geometry"] = clipped
        result.loc[index, "precision_reason"] = ";".join(reasons) if reasons else "corroborated_source_and_sustained_absence"
    # A weak alternative is not evidence of unchanged truth, but it is still
    # evidence of an ambiguous cross-period track. Use the observable long raw
    # candidates here; excluding weak tracks first would hide that ambiguity.
    conflict_networks = {kind: LongitudinalCoverage([axes[i] for i in axes
                         if result.loc[i, "change_typ"] == kind and axes[i].length >= minimum_length
                         and result.loc[i, "paired_valid_ratio"] >= config.minimum_valid_ratio], 3.) for kind in ("added", "removed")}
    for index, axis in axes.items():
        row = result.loc[index]
        opposite = "removed" if row.change_typ == "added" else "added"
        radius = config.opposing_corridor_distance_m
        covered = axis.length-sum(b-a for a, b in conflict_networks[opposite].uncovered(axis, radius))
        ratio = max(0., covered/max(axis.length, 1e-9))
        result.loc[index, "opposing_corridor_ratio"] = ratio
        if covered >= minimum_length and ratio >= config.opposing_corridor_ratio:
            result.loc[index, "publication_state"] = "review"
            result.loc[index, "precision_reason"] += ";opposing_parallel_change_tracks"
        accepted = result.loc[index, "publication_state"] == "accepted"
        result.loc[index, "qa_state"] = ("confirmed" if result.loc[index, "opposite_absent_ratio"] >= .9 and not bool(row.junction)
                                          else "probable") if accepted else "uncertain"
        result.loc[index, "confidence"] = {"confirmed": .85, "probable": .65, "uncertain": .30}[result.loc[index, "qa_state"]]
        result.loc[index, "audit_reason"] += ";"+result.loc[index, "precision_reason"]
    # Restore raw geometry for all review candidates, including corridor conflicts.
    review_mask = result.publication_state == "review"
    result.loc[review_mask, "geometry"] = candidates.reset_index(drop=True).loc[review_mask, "geometry"]
    return result.loc[~review_mask].copy().reset_index(drop=True), result


def annotate_objects(objects, seeds, membership):
    """Publish seed QA lineage without changing the frozen geometry assembler."""
    rank = {"confirmed": 0, "probable": 1, "uncertain": 2}
    objects = objects.copy()
    # Qualification can legitimately publish no objects while retaining review
    # candidates. Keep the QA schema available to writers and empty funnels.
    for key, default in (("qa_state", ""), ("confidence", 0.), ("audit_reason", "")):
        if key not in objects:
            objects[key] = default
    for object_id, group in membership.groupby("object_id") if len(membership) else []:
        local = seeds.iloc[group.seed_id.to_numpy(int)]
        mask = objects.object_id == object_id
        objects.loc[mask, "qa_state"] = max(local.qa_state, key=rank.__getitem__)
        objects.loc[mask, "confidence"] = float(local.confidence.min())
        objects.loc[mask, "audit_reason"] = ";".join(sorted({r for value in local.audit_reason for r in value.split(";")}))
        for state in rank:
            objects.loc[mask, f"{state}_seeds"] = int((local.qa_state == state).sum())
    return objects

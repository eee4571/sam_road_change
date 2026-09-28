"""Conservative, auditable rules. Geometry-only uses identical candidates and guards."""
import copy
import numpy as np
from shapely import union_all
from shapely.geometry import Point
from shapely.ops import substring
from shapely.strtree import STRtree

import bootstrap
from evidence import stations
from graph import candidates
from engine.canonical_road_surface import Edge, _node, _adjacency


def spur_decision(score):
    if score["independent_valid_fraction"] < .9 or score["rgb_valid_fraction"] < .9:
        return "review", "missing_or_nodata_image_evidence"
    if score["independent_support_fraction"] >= .85 and score["bilateral_edge_score"] >= .45:
        return "keep", "sustained_model_and_bilateral_edges"
    if score["cached_width_quality"] is not None and score["cached_width_quality"] >= .16:
        return "keep", "cached_width_observations_support_real_road"
    if score["independent_support_fraction"] <= .15 and score["bilateral_edge_score"] < .2:
        return "delete", "observed_low_model_support_and_absent_bilateral_edges"
    return "review", "conflicting_or_intermediate_evidence"


def gap_decision(score, config):
    if score["independent_valid_fraction"] < .95 or score["rgb_valid_fraction"] < .9:
        return "review", "missing_or_nodata_image_evidence"
    strong = score["probability_mean"] >= config["strong_probability"] or score["molra_surface_support"] >= .9
    if (strong and score["continuous_support_fraction"] >= config["support_fraction"] and
            score["maximum_unsupported_m"] <= config["max_unsupported_m"] and
            score["bilateral_edge_score"] >= .45 and
            score["route_direction_cosine"] >= config["direction_cosine"] and
            score["route_length_ratio"] <= config["max_route_ratio"] and
            score["maximum_turn_degrees"] <= config["max_turn_degrees"]):
        return "connect", "continuous_image_support_and_consistent_direction"
    return "review", "insufficient_continuity_direction_or_curvature_support"


def duplicate_decision(score):
    if score["valid_fraction"] < .9:
        return "review", "missing_ribbon_evidence_or_subpixel_separation"
    if score["two_ribbon_fraction"] >= .7:
        return "keep", "two_supported_ribbons_with_stable_nonroad_separator"
    if score["raw_separator_fraction"] >= .7:
        return "keep", "stable_raw_image_separator_vetoes_merge"
    if score["rgb_valid_fraction"] < .9:
        return "review", "missing_raw_image_ribbon_evidence"
    if score["single_ribbon_fraction"] >= .9 and score["two_ribbon_fraction"] <= .05:
        return "merge", "both_axes_inside_one_continuous_supported_ribbon"
    return "review", "ribbon_structure_ambiguous"


def route_shape(route, candidate):
    _, _, tangent, _ = stations(route, 3.)
    approaches = np.asarray(candidate.geometry["approach_directions"])
    cosine = min(float(tangent[0]@approaches[0]), float(tangent[-1]@approaches[1]))
    turns = np.rad2deg(np.arccos(np.clip(np.sum(tangent[:-1]*tangent[1:], axis=1), -1, 1)))
    return dict(route_direction_cosine=cosine, route_length_ratio=route.length/candidate.axis.length,
                maximum_turn_degrees=float(turns.max(initial=0)))


def safe_gap(route, edges, candidate):
    if not route.is_simple:
        return False
    tree = STRtree([e.axis for e in edges])
    ends = union_all([Point(route.coords[0]).buffer(.05), Point(route.coords[-1]).buffer(.05)])
    for i in tree.query(route, predicate="dwithin", distance=.1):
        if not route.intersection(edges[i].axis.buffer(.025)).difference(ends).is_empty:
            return False
    return True


def safe_duplicate(candidate, edges):
    i, j = candidate.ids
    if candidate.geometry.get("partial_overlap"):
        return False
    nodes = _adjacency(edges, set(range(len(edges))))
    # V1 removes only a redundant arm without stranding any branch contacts.
    return all(len(nodes[n]) == 1 or n in edges[j].nodes for n in edges[i].nodes)


def run(edges, evidence, roi, config, mode, stages=("spur", "gap", "duplicate")):
    edges = copy.deepcopy(edges)
    audit, changes = [], []
    focus = None
    seen = set()
    rounds = config["rounds"] if len(stages) > 1 else 1
    for round_index in range(rounds):
        round_changes = []
        for stage in stages:
            proposals = candidates(edges, roi, config, (stage,), focus)
            removed, added, touched, used_tips = set(), [], set(), set()
            for c in proposals:
                fingerprint = (c.key, tuple(e.axis.wkb for e in (edges[i] for i in c.ids)))
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                score = None
                curve = c.axis
                if c.kind == "spur":
                    # Do not let the supported parent road hide a false terminal arm.
                    score = evidence.measure(substring(c.axis, 0, c.axis.length*.65), c.width)
                    decision, reason = spur_decision(score)
                elif c.kind == "gap":
                    routed = evidence.route(c.axis, c.width) if mode == "image_guided" else c.axis
                    if routed is not None:
                        curve = routed
                    score = evidence.measure(curve, c.width)
                    score.update(route_shape(curve, c))
                    decision, reason = gap_decision(score, config)
                    if routed is None:
                        decision, reason = "review", "no_continuous_evidence_route"
                else:
                    score = evidence.ribbons(edges[c.ids[0]].axis, edges[c.ids[1]].axis)
                    decision, reason = duplicate_decision(score)
                evidence_decision = decision
                if mode == "geometry_only":
                    decision = dict(spur="delete", gap="connect", duplicate="merge")[stage]
                    reason = "geometry_ablation_without_image_veto"
                if touched.intersection(c.ids):
                    decision, reason = "review", "candidate_conflicts_with_accepted_edit"
                if decision == "connect":
                    tips = [tuple(n) for n in c.geometry["tip_nodes"]]
                    if any(n in used_tips for n in tips) or not safe_gap(curve, edges, c):
                        decision, reason = "review", "crossing_contact_or_endpoint_conflict"
                if decision == "merge" and not safe_duplicate(c, edges):
                    decision, reason = "review", "partial_overlap_or_branch_contact_requires_later_rewire"
                if reason in ("candidate_conflicts_with_accepted_edit", "crossing_contact_or_endpoint_conflict",
                              "partial_overlap_or_branch_contact_requires_later_rewire"):
                    # A neighbouring accepted edit can make this candidate safe next round.
                    seen.discard(fingerprint)
                row = dict(candidate_id=c.key, type=c.kind, round=round_index+1, mode=mode,
                           source_ids=sorted({s for i in c.ids for s in edges[i].sources}),
                           geometry_scores=c.geometry, evidence=score, evidence_decision=evidence_decision,
                           decision=decision, reason=reason, candidate_wkt=c.axis.wkt, result_wkt=curve.wkt)
                row["evidence_score"] = score.get("evidence_score", score.get("single_ribbon_fraction"))
                if c.kind == "duplicate":
                    row["comparison_wkt"] = edges[c.ids[1]].axis.wkt
                audit.append(row)
                if decision in ("delete", "merge", "connect"):
                    if decision == "connect":
                        level = edges[c.ids[0]].level
                        added.append(Edge(curve, np.array([0., curve.length]), np.array([c.width]*2),
                                          np.zeros(2), (), level,
                                          (_node(level, curve.coords[0]), _node(level, curve.coords[-1]))))
                        used_tips.update(tuple(n) for n in c.geometry["tip_nodes"])
                    else:
                        removed.update(c.ids if decision == "delete" else (c.ids[0],))
                    touched.update(c.ids)
                    delta = dict(candidate_id=c.key, decision=decision, geometry=curve,
                                 reference=edges[c.ids[1]].axis if decision == "merge" else None)
                    changes.append(delta)
                    round_changes.append(curve)
            # Accepted edits cannot introduce interior crossings. Preserve all
            # unaffected noded edges and cached widths, update adjacency only.
            edges = [e for i, e in enumerate(edges) if i not in removed] + added
        if not round_changes:
            break
        focus = union_all(round_changes).buffer(config["halo_m"])
    return edges, audit, changes

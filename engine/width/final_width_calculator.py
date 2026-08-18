from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from molra_centerline_width import build_width_change_segments, sample_widths_by_normal


@dataclass(frozen=True)
class FinalWidthConfig:
    pixel_size: float = 1.0
    sample_step_px: float = 5.0
    normal_step_px: float = 1.0
    max_search_px: float = 120.0
    snap_radius_px: int = 8
    junction_buffer_px: float = 30.0
    border_margin_px: int = 2
    max_snap_distance_px: float = 4.0
    max_asymmetry_ratio: float = 0.65
    width_change_ratio: float = 0.35
    width_change_min_samples: int = 3
    min_edge_coverage: float = 0.6
    short_gap_px: float = 80.0
    max_width_cv: float = 0.5
    outlier_mad_scale: float = 3.5
    hybrid_agreement_ratio: float = 0.35
    junction_width_factor: float = 1.75
    neighbor_direction_cosine: float = 0.80


@dataclass(frozen=True)
class FinalWidthRequest:
    nodes_rc: np.ndarray
    edges: np.ndarray
    road_surface: np.ndarray | None
    config: FinalWidthConfig
    edge_metadata: tuple[dict, ...] = field(default_factory=tuple)


@dataclass
class FinalWidthResult:
    edge_widths: list[dict]
    samples: list[dict]
    segments: list[dict]
    algorithm: str = "normal_boundary_v1"
    metadata: dict = field(default_factory=dict)


FinalWidthCalculator = Callable[[FinalWidthRequest], FinalWidthResult]


def calculate_final_widths(request: FinalWidthRequest) -> FinalWidthResult:
    """Default final-width implementation; replace through the public interface."""
    edge_count = int(request.edges.shape[0])
    if request.road_surface is None or edge_count == 0:
        source = "unresolved_no_final_surface_mask" if request.road_surface is None else "unresolved_empty_final_graph"
        return FinalWidthResult(
            edge_widths=[
                {"edge_id": edge_id, "width_px": 0.0, "width_units": 0.0, "source": source, "status": "width_unresolved"}
                for edge_id in range(edge_count)
            ],
            samples=[],
            segments=[],
            metadata={"measured_edge_count": 0, "unresolved_edge_count": edge_count},
        )

    config = request.config
    samples = sample_widths_by_normal(
        request.nodes_rc,
        request.edges,
        request.road_surface.astype(np.uint8),
        sample_step_px=config.sample_step_px,
        normal_step_px=config.normal_step_px,
        max_search_px=config.max_search_px,
        pixel_size=config.pixel_size,
        snap_radius_px=config.snap_radius_px,
        junction_buffer_px=config.junction_buffer_px,
        border_margin_px=config.border_margin_px,
        max_snap_distance_px=config.max_snap_distance_px,
        max_asymmetry_ratio=config.max_asymmetry_ratio,
    )
    widths_by_edge: dict[int, list[float]] = {}
    for sample in samples:
        width = float(sample.get("width_px", 0.0) or 0.0)
        if width > 0:
            widths_by_edge.setdefault(int(sample["edge_id"]), []).append(width)

    adjacency: list[list[int]] = [[] for _ in range(len(request.nodes_rc))]
    for edge_id, (src_idx, dst_idx) in enumerate(request.edges.tolist()):
        adjacency[int(src_idx)].append(edge_id)
        adjacency[int(dst_idx)].append(edge_id)
    degrees = np.asarray([len(items) for items in adjacency], dtype=np.int32)
    junction_nodes = np.where(degrees >= 3)[0]
    junction_near_node = np.zeros(len(request.nodes_rc), dtype=bool)
    if junction_nodes.size:
        distances = np.linalg.norm(
            request.nodes_rc[:, None, :] - request.nodes_rc[junction_nodes][None, :, :], axis=2
        )
        junction_near_node = np.min(distances, axis=1) <= float(config.junction_buffer_px)

    edge_widths = []
    for edge_id in range(edge_count):
        widths = widths_by_edge.get(edge_id, [])
        if widths:
            width_px = float(np.median(np.asarray(widths, dtype=np.float32)))
            source = "remeasured_on_final_graph"
            status = "measured_on_final_graph"
        else:
            width_px = 0.0
            source = "unresolved_after_final_measurement"
            status = "width_unresolved"
        edge_widths.append(
            {
                "edge_id": edge_id,
                "width_px": width_px,
                "width_units": width_px * config.pixel_size,
                "source": source,
                "status": status,
                "quality_grade": "A" if widths else "C",
            }
        )

    # A valid normal probe beside a degree-2 split can still cross the whole
    # junction lobe.  Reject only extreme local spikes when a neighbouring
    # branch supplies contradictory measured evidence; ordinary width changes
    # outside the junction buffer remain untouched.
    rejected_junction_outliers = 0
    directions = np.zeros((edge_count, 2), dtype=np.float32)
    lengths = np.zeros(edge_count, dtype=np.float32)
    for edge_id, (src_idx, dst_idx) in enumerate(request.edges.tolist()):
        vector = request.nodes_rc[int(dst_idx)] - request.nodes_rc[int(src_idx)]
        length = float(np.linalg.norm(vector))
        lengths[edge_id] = length
        if length > 1e-6:
            directions[edge_id] = vector / length
    for edge_id, (src_idx, dst_idx) in enumerate(request.edges.tolist()):
        row = edge_widths[edge_id]
        width = float(row["width_px"])
        if (
            width <= 0
            or lengths[edge_id] > 2.0 * float(config.junction_buffer_px)
            or not (junction_near_node[int(src_idx)] or junction_near_node[int(dst_idx)])
        ):
            continue
        neighbours: list[float] = []
        for node_id in (int(src_idx), int(dst_idx)):
            for other_id in adjacency[node_id]:
                if other_id == edge_id:
                    continue
                other = edge_widths[other_id]
                other_width = float(other["width_px"])
                if other_width <= 0 or str(other.get("quality_grade", "C")) == "C":
                    continue
                cosine = abs(float(np.dot(directions[edge_id], directions[other_id])))
                if cosine >= float(config.neighbor_direction_cosine):
                    neighbours.append(other_width)
        if not neighbours:
            continue
        neighbour_width = float(np.median(np.asarray(neighbours, dtype=np.float32)))
        outlier_limit = max(
            neighbour_width * max(2.5, float(config.junction_width_factor)),
            neighbour_width + 20.0,
        )
        if width > outlier_limit:
            row.update({
                "width_px": 0.0,
                "width_units": 0.0,
                "source": "junction_outlier_rejected",
                "status": "width_unresolved_junction_outlier",
                "quality_grade": "C",
            })
            rejected_junction_outliers += 1

    segments = build_width_change_segments(
        samples,
        pixel_size=config.pixel_size,
        change_ratio=config.width_change_ratio,
        min_samples=max(1, config.width_change_min_samples),
    )
    measured_count = sum(row["width_px"] > 0 for row in edge_widths)
    return FinalWidthResult(
        edge_widths=edge_widths,
        samples=samples,
        segments=segments,
        metadata={
            "measured_edge_count": measured_count,
            "unresolved_edge_count": edge_count - measured_count,
            "junction_width_outlier_rejected_edge_count": rejected_junction_outliers,
        },
    )


def load_width_calculator(spec: str = "") -> FinalWidthCalculator:
    """Load `module:function`; an empty spec selects the built-in calculator."""
    if not spec:
        return calculate_final_widths
    if ":" not in spec:
        raise ValueError("Width calculator must use the form module:function")
    module_name, function_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if not module_name.startswith("sam_width_experiment."):
            raise
        tool_dir = Path(__file__).resolve().parent
        if str(tool_dir) not in sys.path:
            sys.path.insert(0, str(tool_dir))
        module = importlib.import_module(module_name.split(".", 1)[1])
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(f"Width calculator is not callable: {spec}")
    return function


def validate_width_result(result: FinalWidthResult, edge_count: int) -> None:
    if not isinstance(result, FinalWidthResult):
        raise TypeError("Width calculator must return FinalWidthResult")
    edge_ids = [int(row.get("edge_id", -1)) for row in result.edge_widths]
    if edge_ids != list(range(edge_count)):
        raise ValueError(f"Width calculator must return exactly one ordered result per final edge: {edge_ids}")
    for row in result.edge_widths:
        width = float(row.get("width_px", 0.0) or 0.0)
        if not np.isfinite(width) or width < 0:
            raise ValueError(f"Invalid final width for edge {row.get('edge_id')}: {width}")

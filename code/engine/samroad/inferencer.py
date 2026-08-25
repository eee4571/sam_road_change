import numpy as np
import os
import csv
import json
import imageio
import torch
import cv2
import re
from utils import load_config, create_output_dir_and_save_config
from dataset import (
    get_cityscale_split_from_config,
    get_eval_patches_per_axis,
    get_patch_info_one_img,
    globalscale_data_partition,
    read_rgb_img,
    spacenet_data_partition,
)
from modelinfer import SAMRoadplus
import graph_extraction
import graph_utils
import triage
import pickle
import scipy
import rtree
from collections import defaultdict
import time
import os
import copy
from argparse import ArgumentParser
from pathlib import Path
from package_paths import INFER_RUNS_ROOT, PROJECT_ROOT, resolve_path
from input_catalog import read_path_list
from image_resume import (
    ImageResumeManager,
    build_batch_identity,
    effective_config_identity,
    ensure_unique_output_stems,
    marker_summaries,
)
from fast_probability import build_fast_enhanced_road_probability

try:
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None


def resolve_repo_path(path):
    return resolve_path(path, PROJECT_ROOT)



parser = ArgumentParser()
parser.add_argument(
    "--checkpoint", default='', help="checkpoint of the model to test."
)
parser.add_argument(
    "--junction_node_mode", choices=["sparse", "dense_legacy"], default="",
    help="Override config junction sampling mode.",
)
parser.add_argument(
    "--config", default="", help="model config."
)
parser.add_argument(
    "--output_dir", default="", help="Name of the output dir, if not specified will use timestamp"
)
parser.add_argument(
    "--input_dir", default="", help="Directory containing input images for inference"
)
parser.add_argument(
    "--input_txt_dir", default="", help="Directory containing multiple txt files with image paths"
)
parser.add_argument(
    "--output_root", default=str(INFER_RUNS_ROOT), help="Directory where inference batches are written"
)
parser.add_argument("--input_gsd", type=float, default=None, help="Input image GSD in meters/pixel. Overrides config and GeoTIFF metadata.")
parser.add_argument("--model_gsd", type=float, default=None, help="Model/training GSD in meters/pixel. Overrides config MODEL_GSD.")
parser.add_argument(
    "--rescale_to_model_gsd",
    choices=["on", "off"],
    default="off",
    help="Enable or disable rescaling input images to MODEL_GSD for this inference run.",
)
parser.add_argument("--device", default="cuda", help="device to use for training")
parser.add_argument(
    "--max_image_megapixels",
    type=float,
    default=25.0,
    help="Fail fast above this single-image size to avoid exhausting RAM/VRAM. Set 0 to disable.",
)
parser.add_argument(
    "--resume-existing-images", action="store_true",
    help="Validate and skip complete input images in an interrupted batch.",
)
parser.add_argument(
    "--pipeline-state", default="",
    help="Optional pipeline job_state.json used only for safe legacy-output adoption.",
)
parser.add_argument(
    "--execution-profile", choices=["full", "fast"], default="full",
    help="Fast writes road probability plus native TopoNet and skips weak postprocess.",
)
args = parser.parse_args()


def resolve_torch_device(device_arg):
    device_name = str(device_arg).strip().lower()
    if device_name in {"gpu", "cuda"}:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU inference was requested, but CUDA is not available. "
                "Please rerun and choose CPU, or install a CUDA-enabled PyTorch environment."
            )
        return "cuda", torch.device("cuda")
    if device_name == "auto":
        if torch.cuda.is_available():
            return "cuda", torch.device("cuda")
        return "cpu", torch.device("cpu")
    if device_name == "cpu":
        return "cpu", torch.device("cpu")
    raise ValueError(f"Unsupported device '{device_arg}'. Use cuda/gpu, cpu, or auto.")


def get_img_paths(root_dir, image_indices):
    img_paths = []

    for ind in image_indices:
        img_paths.append(os.path.join(root_dir, f"region_{ind}_sat.png"))
    return img_paths


def _strip_txt_line(line):
    line = line.strip().strip("\ufeff")
    if not line or line.startswith("#"):
        return ""
    if line[0] in {'"', "'"} and line[-1:] == line[0]:
        line = line[1:-1].strip()
    return line


def load_image_paths_from_txt(txt_path):
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Input txt does not exist: {txt_path}")
    listing = read_path_list(txt_path, search_roots=(PROJECT_ROOT,))
    return [entry.path for entry in listing.entries]


def list_txt_files(input_txt_dir):
    input_txt_dir = Path(input_txt_dir)
    if not input_txt_dir.exists():
        raise FileNotFoundError(f"Input txt dir does not exist: {input_txt_dir}")
    return sorted(path for path in input_txt_dir.rglob("*.txt") if path.is_file())


def list_input_images(input_dir):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input image directory does not exist: {input_dir}")
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in valid_suffixes
    )


def gather_inference_inputs(input_dir="", input_txt_dir=""):
    sources = [bool(str(input_dir).strip()), bool(str(input_txt_dir).strip())]
    if sum(sources) != 1:
        raise ValueError("Please specify exactly one of --input_dir or --input_txt_dir.")

    if str(input_dir).strip():
        return [("dir", Path(input_dir), list_input_images(resolve_repo_path(input_dir)))]

    root_txt_dir = resolve_repo_path(input_txt_dir)
    batches = []
    for txt_file in list_txt_files(root_txt_dir):
        rel_name = txt_file.relative_to(root_txt_dir).with_suffix("")
        source_name = "__".join(rel_name.parts)
        batches.append((source_name, txt_file, load_image_paths_from_txt(txt_file)))
    return batches


def _synchronize_cuda_for_timing():
    """Synchronize only at coarse per-image timing boundaries."""
    if str(args.device).lower().startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def resolve_relative_context_for_postprocess(
    road_mask,
    config,
    *,
    scene_state,
    distance_scale=1.0,
    precomputed_context=None,
):
    """Return a shape-compatible Relative context and the actual call count."""
    expected_shape = np.asarray(road_mask).shape
    if precomputed_context is not None:
        relative_score = np.asarray(precomputed_context.get("relative_score", []))
        if relative_score.shape == expected_shape:
            return precomputed_context, 0
    return (
        graph_extraction.compute_relative_roadness(
            road_mask,
            config,
            scene_state=scene_state,
            distance_scale=distance_scale,
        ),
        1,
    )


def run_inference_on_images(
    net, config, input_img_paths, output_dir, input_label, *,
    resume_existing_images=False, resume_identity=None,
    legacy_metadata=None, pipeline_state=None,
):
    total_inference_seconds = 0.0
    recovery_summaries = []
    profile_decisions = []
    ensure_unique_output_stems(input_img_paths)
    resume_manager = (
        ImageResumeManager(
            output_dir, resume_identity, enabled=resume_existing_images,
            legacy_metadata=legacy_metadata, pipeline_state=pipeline_state,
        )
        if isinstance(resume_identity, dict) else None
    )
    resume_counts = {
        "validated_marker_skip_count": 0,
        "legacy_adopted_count": 0,
        "inferred_count": 0,
    }
    print(f'Found {len(input_img_paths)} image(s) under {input_label}.')
    print(f'Inference patch size: {config.PATCH_SIZE}x{config.PATCH_SIZE}')
    print(f'Inference device: {resolved_device_name}')
    print(
        f'Image resume: {"enabled" if resume_existing_images else "disabled"}; '
        f'total image count: {len(input_img_paths)}.'
    )

    for img_path in input_img_paths:
        # Threshold selection is per complete image.  Never mutate the shared
        # batch config, otherwise one weak image would leak into the next tile.
        image_config = copy.deepcopy(config)
        if args.execution_profile == "fast":
            image_config.RELATIVE_ROADNESS_ENABLED = False
            image_config.RELATIVE_INJECT_INTO_TOPONET = False
        img_path = Path(img_path).expanduser().resolve()
        img_id = img_path.stem
        if resume_manager is not None:
            decision = resume_manager.inspect(img_path)
            if decision["action"] == "skip":
                recovery_summary, profile_decision = marker_summaries(decision["marker"])
                recovery_summaries.append(recovery_summary)
                profile_decisions.append(profile_decision)
                total_inference_seconds += float(recovery_summary.get("total_image_seconds", 0.0) or 0.0)
                if decision["origin"] == "legacy_adopted":
                    resume_counts["legacy_adopted_count"] += 1
                    print(f'[image-resume] Adopted complete legacy outputs for {img_path}.')
                else:
                    resume_counts["validated_marker_skip_count"] += 1
                    print(f'[image-resume] Validated and skipped {img_path}.')
                continue
            print(f'[image-resume] Reprocessing {img_path}: {decision.get("reason", "not complete")}')
            resume_manager.prepare_for_processing(img_path)
        resume_counts["inferred_count"] += 1
        print(f'Processing {img_path}')
        img = read_rgb_img(img_path)
        original_height, original_width = img.shape[:2]
        image_megapixels = original_height * original_width / 1_000_000.0
        if args.max_image_megapixels > 0 and image_megapixels > args.max_image_megapixels:
            raise RuntimeError(
                f"Input image is {original_width}x{original_height} ({image_megapixels:.1f} MP), "
                f"above the safety limit of {args.max_image_megapixels:g} MP. "
                "Split it into smaller georeferenced tiles, or set --max_image_megapixels 0 only if sufficient RAM is available."
            )
        infer_img, resize_factor, input_gsd, model_gsd = rescale_image_to_model_gsd(img, img_path, config)
        if resize_factor != 1.0:
            print(
                f'  Rescaled input from {original_width}x{original_height} to '
                f'{infer_img.shape[1]}x{infer_img.shape[0]} for model GSD '
                f'{model_gsd:g}m/pixel (input GSD {input_gsd:g}m/pixel).'
            )
        else:
            print(f'  Inference GSD scale unchanged (input GSD {input_gsd:g}m/pixel, model GSD {model_gsd:g}m/pixel).')

        infer_height, infer_width = infer_img.shape[:2]
        padded_img = pad_image_to_min_size(infer_img, config.PATCH_SIZE)
        padded_height, padded_width = padded_img.shape[:2]
        if (padded_height, padded_width) != (infer_height, infer_width):
            print(
                f'  Input image is smaller than patch size; padded from '
                f'{infer_width}x{infer_height} to {padded_width}x{padded_height}.'
            )
        elif infer_height > config.PATCH_SIZE or infer_width > config.PATCH_SIZE:
            print('  Input image is larger than patch size; running sliding-window inference.')

        image_start_seconds = time.perf_counter()
        (
            pred_nodes,
            pred_edges,
            edge_confidences,
            candidate_edges,
            candidate_confidences,
            itsc_mask,
            road_mask,
            profile_decision,
            precomputed_relative_context,
            performance_summary,
        ) = infer_one_img(
            net, padded_img, image_config,
            diagnostic_shape=(infer_height, infer_width),
        )

        itsc_mask = itsc_mask[:infer_height, :infer_width]
        road_mask = road_mask[:infer_height, :infer_width]
        fast_enhanced_mask = None
        if args.execution_profile == "fast" and isinstance(precomputed_relative_context, dict):
            fast_enhanced_mask = np.asarray(
                precomputed_relative_context.get("enhanced_road_mask"), dtype=np.uint8,
            )[:infer_height, :infer_width]
        candidate_nodes, candidate_edges, candidate_confidences = filter_graph_to_image_bounds(
            pred_nodes, candidate_edges, infer_height, infer_width, candidate_confidences
        )
        pred_nodes, pred_edges, edge_confidences = filter_graph_to_image_bounds(
            pred_nodes, pred_edges, infer_height, infer_width, edge_confidences
        )
        if resize_factor != 1.0:
            itsc_mask = cv2.resize(itsc_mask, (original_width, original_height), interpolation=cv2.INTER_AREA)
            road_mask = cv2.resize(road_mask, (original_width, original_height), interpolation=cv2.INTER_AREA)
            if fast_enhanced_mask is not None:
                fast_enhanced_mask = cv2.resize(
                    fast_enhanced_mask,
                    (original_width, original_height),
                    interpolation=cv2.INTER_AREA,
                )
            pred_nodes = pred_nodes.astype(np.float32) / float(resize_factor)
            candidate_nodes = candidate_nodes.astype(np.float32) / float(resize_factor)
            candidate_nodes, candidate_edges, candidate_confidences = filter_graph_to_image_bounds(
                candidate_nodes, candidate_edges, original_height, original_width, candidate_confidences
            )
            pred_nodes, pred_edges, edge_confidences = filter_graph_to_image_bounds(
                pred_nodes, pred_edges, original_height, original_width, edge_confidences
            )

        if args.execution_profile == "fast":
            performance_summary["total_image_seconds"] = float(
                time.perf_counter() - image_start_seconds
            )
            total_inference_seconds += performance_summary["total_image_seconds"]
            profile_decision.update({
                "graph_extraction_skipped": False,
                "toponet_skipped": False,
                "weak_postprocess_skipped": True,
            })
            profile_decision = {"image": str(img_path), "tile": img_id, **profile_decision}
            recovery_summary = {
                "tile": img_id,
                **profile_decision,
                "execution_profile": "fast",
                "centerline_method": "native_toponet_on_fast_enhanced_probability",
                **performance_summary,
            }
            profile_decisions.append(profile_decision)
            recovery_summaries.append(recovery_summary)
            mask_save_dir = os.path.join(output_dir, "mask")
            os.makedirs(mask_save_dir, exist_ok=True)
            probability_path = os.path.join(mask_save_dir, f"{img_id}_road.png")
            if not cv2.imwrite(probability_path, road_mask):
                raise OSError(f"Cannot write Fast road probability: {probability_path}")
            enhanced_path = os.path.join(mask_save_dir, f"{img_id}_fast_enhanced.png")
            if not cv2.imwrite(
                enhanced_path,
                fast_enhanced_mask if fast_enhanced_mask is not None else road_mask,
            ):
                raise OSError(f"Cannot write enhanced Fast road probability: {enhanced_path}")
            boost_path = os.path.join(mask_save_dir, f"{img_id}_fast_boost.png")
            boost_mask = np.clip(
                (fast_enhanced_mask if fast_enhanced_mask is not None else road_mask).astype(np.int16)
                - road_mask.astype(np.int16),
                0,
                255,
            ).astype(np.uint8)
            if not cv2.imwrite(boost_path, boost_mask):
                raise OSError(f"Cannot write Fast probability boost: {boost_path}")
            graph_save_dir = os.path.join(output_dir, "graph")
            os.makedirs(graph_save_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(graph_save_dir, f"{img_id}_fast_topology.npz"),
                nodes=np.asarray(pred_nodes, dtype=np.float32),
                edges=np.asarray(pred_edges, dtype=np.int32).reshape(-1, 2),
                scores=np.asarray(edge_confidences, dtype=np.float32),
            )
            if resume_manager is not None:
                resume_manager.complete(img_path, recovery_summary, profile_decision)
            print(
                f"Done Fast probability + native TopoNet for {img_id}: "
                f"fast_graph_point_count="
                f"{performance_summary['fast_graph_point_count']}, "
                f"toponet_candidate_edge_count="
                f"{performance_summary['toponet_candidate_edge_count']}, "
                f"toponet_final_edge_count="
                f"{performance_summary['toponet_final_edge_count']}."
            )
            continue

        postprocess_distance_scale = 1.0 / max(float(resize_factor), 1e-6)
        bootstrap_candidate_audit = []
        weak_start_seconds = time.perf_counter()
        relative_start_seconds = time.perf_counter()
        relative_context, additional_relative_calls = resolve_relative_context_for_postprocess(
            road_mask,
            image_config,
            scene_state=profile_decision["scene_confidence_state"],
            distance_scale=postprocess_distance_scale,
            precomputed_context=precomputed_relative_context,
        )
        performance_summary["relative_roadness_seconds"] += float(
            time.perf_counter() - relative_start_seconds
        )
        performance_summary["relative_compute_call_count"] += int(additional_relative_calls)
        relative_context["diagnostics"].update({
            "relative_injected_into_toponet": performance_summary[
                "relative_injected_into_toponet"
            ],
        })
        profile_decision.update(relative_context.get("diagnostics", {}))
        pred_nodes, pred_edges, edge_metadata, recovery_summary = graph_extraction.postprocess_weak_road_network(
            pred_nodes,
            pred_edges,
            road_mask,
            image_config,
            edge_scores=edge_confidences,
            distance_scale=postprocess_distance_scale,
            relative_context=relative_context,
            bootstrap_candidate_audit=bootstrap_candidate_audit,
            topology_candidate_nodes_rc=candidate_nodes,
            topology_candidate_edges=candidate_edges,
            topology_candidate_scores=candidate_confidences,
        )
        performance_summary["weak_postprocess_seconds"] = float(
            time.perf_counter() - weak_start_seconds
        )
        weak_phase_timing = recovery_summary.get("timing", {})
        for phase_name, output_name in {
            "diagnosis_seconds": "weak_diagnosis_seconds",
            "relative_context_seconds": "weak_relative_context_seconds",
            "bootstrap_seconds": "weak_bootstrap_seconds",
            "weak_endpoint_recovery_seconds": "weak_endpoint_recovery_seconds",
            "endpoint_to_segment_recovery_seconds": "endpoint_to_segment_recovery_seconds",
            "connectivity_statistics_seconds": "weak_connectivity_statistics_seconds",
        }.items():
            performance_summary[output_name] = float(
                weak_phase_timing.get(phase_name, 0.0)
            )
        performance_summary.update({
            "relative_graph_point_count": int(
                recovery_summary.get("relative_graph_point_count", 0)
            ),
            "toponet_candidate_edge_count": int(candidate_edges.shape[0]),
            "toponet_pred_edge_count": int(sum(
                float(score) > float(image_config.TOPO_THRESHOLD)
                for score in candidate_confidences.tolist()
            )),
            "relative_final_centerline_length": float(
                recovery_summary.get("relative_final_length_px", 0.0)
            ),
        })
        performance_summary["total_image_seconds"] = float(
            time.perf_counter() - image_start_seconds
        )
        total_inference_seconds += performance_summary["total_image_seconds"]
        edge_confidences = np.asarray(
            [row["topology_probability"] for row in edge_metadata], dtype=np.float32
        )
        profile_decision = {"image": str(img_path), "tile": img_id, **profile_decision}
        profile_decisions.append(profile_decision)
        recovery_summary = {
            "tile": img_id,
            **profile_decision,
            **recovery_summary,
            **performance_summary,
            "requested_profile": profile_decision["requested_profile"],
            "effective_profile": profile_decision["effective_profile"],
        }
        recovery_summaries.append(recovery_summary)

        viz_img = np.copy(img)
        mask_save_dir = os.path.join(output_dir, 'mask')
        os.makedirs(mask_save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(mask_save_dir, f'{img_id}_road.png'), road_mask)
        cv2.imwrite(os.path.join(mask_save_dir, f'{img_id}_itsc.png'), itsc_mask)
        cv2.imwrite(os.path.join(mask_save_dir, f'{img_id}_centerline_probability.png'), road_mask)
        cv2.imwrite(
            os.path.join(mask_save_dir, f'{img_id}_relative_roadness.png'),
            np.clip(relative_context['relative_score'] * 255.0, 0, 255).astype(np.uint8),
        )
        cv2.imwrite(
            os.path.join(mask_save_dir, f'{img_id}_relative_candidate.png'),
            relative_context['relative_candidate_mask'].astype(np.uint8) * 255,
        )
        for mask_name in (
            'relative_skeleton_raw', 'relative_skeleton_normalized',
            'junction_zone_mask', 'pruned_spur_mask', 'collapsed_zone_mask',
        ):
            cv2.imwrite(
                os.path.join(mask_save_dir, f'{img_id}_{mask_name}.png'),
                np.asarray(relative_context[mask_name], dtype=np.uint8) * 255,
            )
        high_threshold, _low_threshold, _profile = graph_extraction.resolve_road_thresholds(image_config)
        combined_candidate = (
            (graph_extraction._probability01(road_mask) >= high_threshold)
            | (relative_context['relative_candidate_mask'] > 0)
        )
        cv2.imwrite(
            os.path.join(mask_save_dir, f'{img_id}_combined_candidate.png'),
            combined_candidate.astype(np.uint8) * 255,
        )

        viz_save_dir = os.path.join(output_dir, 'viz')
        os.makedirs(viz_save_dir, exist_ok=True)
        norm_scale = np.array([[viz_img.shape[0], viz_img.shape[1]]], dtype=np.float32)
        viz_img = triage.visualize_image_and_graph(viz_img, pred_nodes / norm_scale, pred_edges, viz_img.shape[0])
        cv2.imwrite(os.path.join(viz_save_dir, f'{img_id}.png'), viz_img)
        probability_color = cv2.applyColorMap(road_mask, cv2.COLORMAP_VIRIDIS)
        relative_color = cv2.applyColorMap(
            np.clip(relative_context['relative_score'] * 255.0, 0, 255).astype(np.uint8),
            cv2.COLORMAP_VIRIDIS,
        )
        relative_color[relative_context['relative_candidate_mask'] > 0] = (0, 165, 255)
        chain_panel = np.copy(img)
        chain_panel[relative_context['relative_skeleton'] > 0] = (0, 255, 0)
        acceptance_overlay = np.copy(img)
        rejected_structure = np.asarray(relative_context.get('relative_rejected_skeleton', []))
        if rejected_structure.shape == acceptance_overlay.shape[:2]:
            acceptance_overlay[rejected_structure > 0] = (0, 0, 255)
        decision_colors = {
            'auto': (0, 255, 0),
            'review': (0, 165, 255),
            'rejected': (0, 0, 255),
        }
        for row in bootstrap_candidate_audit:
            if row.get('candidate_source') not in {'relative', 'absolute+relative'}:
                continue
            path = np.asarray(row.get('path', []), dtype=np.int32).reshape(-1, 2)
            if len(path) < 2:
                continue
            points = path[:, ::-1].reshape(-1, 1, 2)
            cv2.polylines(
                acceptance_overlay,
                [points],
                False,
                decision_colors.get(row.get('decision', row.get('qa_state')), (0, 0, 255)),
                3,
                cv2.LINE_AA,
            )
        final_colored = np.copy(img)
        for edge_id, (src_idx, dst_idx) in enumerate(pred_edges.tolist()):
            row = edge_metadata[edge_id]
            relative_edge = (
                row.get('candidate_source') in {'relative', 'absolute+relative'}
                or str(row.get('line_source', '')).startswith('relative')
            )
            color = (0, 255, 0) if relative_edge else (0, 220, 255)
            src = pred_nodes[src_idx]
            dst = pred_nodes[dst_idx]
            cv2.line(
                final_colored,
                (int(round(src[1])), int(round(src[0]))),
                (int(round(dst[1])), int(round(dst[0]))),
                color,
                3,
                cv2.LINE_AA,
            )
        compare_panels = [
            img, probability_color, relative_color,
            chain_panel, acceptance_overlay, final_colored,
        ]
        compare_labels = [
            'image', 'raw probability', 'relative candidate',
            'relative chain', 'auto / review / rejected', 'final vector',
        ]
        max_panel_width = 720
        if original_width > max_panel_width:
            panel_scale = max_panel_width / float(original_width)
            compare_panels = [
                cv2.resize(
                    panel,
                    (max_panel_width, max(1, int(round(original_height * panel_scale)))),
                    interpolation=cv2.INTER_AREA,
                )
                for panel in compare_panels
            ]
        for panel, label in zip(compare_panels, compare_labels):
            cv2.rectangle(panel, (0, 0), (220, 30), (255, 255, 255), -1)
            cv2.putText(panel, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.imwrite(
            os.path.join(viz_save_dir, f'{img_id}_relative_compare.png'),
            np.concatenate([
                np.concatenate(compare_panels[:3], axis=1),
                np.concatenate(compare_panels[3:], axis=1),
            ], axis=0),
        )
        cv2.imwrite(
            os.path.join(viz_save_dir, f'{img_id}_relative_acceptance_overlay.png'),
            acceptance_overlay,
        )

        large_map_sat2graph_format = graph_utils.convert_to_sat2graph_format(pred_nodes, pred_edges)
        graph_save_dir = os.path.join(output_dir, 'graph')
        os.makedirs(graph_save_dir, exist_ok=True)
        graph_save_path = os.path.join(graph_save_dir, f'{img_id}.p')
        with open(graph_save_path, 'wb') as file:
            pickle.dump(large_map_sat2graph_format, file)
        score_path = os.path.join(graph_save_dir, f'{img_id}_edge_scores.csv')
        with open(score_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=[
                'edge_id', 'src_row', 'src_col', 'dst_row', 'dst_col',
                'topology_probability', 'line_source', 'recovery_score',
                'center_conf', 'background_conf', 'probability_contrast',
                'surface_conf', 'recovery_reason', 'qa_state', 'recovery_id',
                'candidate_source', 'scene_rank_mean', 'local_background_mean',
                'local_contrast_mean', 'normalized_contrast_mean',
                'relative_score_mean', 'relative_score_q25', 'relative_fraction',
            ])
            writer.writeheader()
            for edge_id, ((src_idx, dst_idx), score, metadata) in enumerate(
                zip(pred_edges.tolist(), edge_confidences.tolist(), edge_metadata)
            ):
                src_row, src_col = pred_nodes[src_idx]
                dst_row, dst_col = pred_nodes[dst_idx]
                writer.writerow({
                    'edge_id': edge_id,
                    'src_row': float(src_row), 'src_col': float(src_col),
                    'dst_row': float(dst_row), 'dst_col': float(dst_col),
                    'topology_probability': float(score),
                    'line_source': metadata['line_source'],
                    'recovery_score': metadata['recovery_score'],
                    'center_conf': metadata['center_conf'],
                    'background_conf': metadata.get('background_conf', 0.0),
                    'probability_contrast': metadata.get('probability_contrast', 0.0),
                    'surface_conf': metadata['surface_conf'],
                    'recovery_reason': metadata['recovery_reason'],
                    'qa_state': metadata['qa_state'],
                    'recovery_id': metadata['recovery_id'],
                    'candidate_source': metadata.get('candidate_source', 'absolute'),
                    'scene_rank_mean': metadata.get('scene_rank_mean', 0.0),
                    'local_background_mean': metadata.get('local_background_mean', 0.0),
                    'local_contrast_mean': metadata.get('local_contrast_mean', 0.0),
                    'normalized_contrast_mean': metadata.get('normalized_contrast_mean', 0.0),
                    'relative_score_mean': metadata.get('relative_score_mean', 0.0),
                    'relative_score_q25': metadata.get('relative_score_q25', 0.0),
                    'relative_fraction': metadata.get('relative_fraction', 0.0),
                })
        with open(os.path.join(graph_save_dir, f'{img_id}_weak_recovery.json'), 'w', encoding='utf-8') as file:
            json.dump(recovery_summary, file, ensure_ascii=False, indent=2)
        with open(os.path.join(graph_save_dir, f'{img_id}_relative_acceptance_funnel.json'), 'w', encoding='utf-8') as file:
            json.dump(recovery_summary.get('relative_acceptance_funnel', {}), file, ensure_ascii=False, indent=2)
        with open(os.path.join(graph_save_dir, f'{img_id}_relative_skeleton_normalization.json'), 'w', encoding='utf-8') as file:
            json.dump(
                {
                    key: value for key, value in relative_context.get('diagnostics', {}).items()
                    if key.startswith('raw_')
                    or key.startswith('normalized_')
                    or key.startswith('structure_rescued_')
                    or key in {
                        'pruned_spur_count', 'collapsed_zone_count',
                        'junction_zone_radius_px', 'junction_cluster_radius_px',
                        'junction_zones',
                    }
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        review_rows = [
            row for row in bootstrap_candidate_audit
            if row.get('decision') == 'review'
            and row.get('candidate_source') in {'relative', 'absolute+relative'}
        ]
        review_path = os.path.join(graph_save_dir, f'{img_id}_relative_review_candidates.csv')
        review_fields = [
            'candidate_id', 'decision', 'review_reason', 'candidate_source',
            'path_length', 'tortuosity', 'relative_evidence_tier',
            'relative_score_mean', 'relative_score_q25', 'scene_rank_mean',
            'normalized_contrast_mean', 'scale_agreement_mean',
            'connection_count', 'endpoint_alignment',
            'topology_candidate_support_fraction', 'path',
        ]
        with open(review_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=review_fields)
            writer.writeheader()
            for candidate_id, row in enumerate(review_rows):
                writer.writerow({
                    key: (
                        candidate_id if key == 'candidate_id'
                        else json.dumps(row.get(key, []), ensure_ascii=False)
                        if key == 'path'
                        else row.get(key, '')
                    )
                    for key in review_fields
                })
        candidate_path = os.path.join(graph_save_dir, f'{img_id}_edge_candidates.csv')
        with open(candidate_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(
                file,
                fieldnames=['candidate_id', 'src_row', 'src_col', 'dst_row', 'dst_col', 'topology_probability', 'selected'],
            )
            writer.writeheader()
            for candidate_id, ((src_idx, dst_idx), score) in enumerate(
                zip(candidate_edges.tolist(), candidate_confidences.tolist())
            ):
                src_row, src_col = candidate_nodes[src_idx]
                dst_row, dst_col = candidate_nodes[dst_idx]
                writer.writerow({
                    'candidate_id': candidate_id,
                    'src_row': float(src_row), 'src_col': float(src_col),
                    'dst_row': float(dst_row), 'dst_col': float(dst_col),
                    'topology_probability': float(score),
                    'selected': int(float(score) > float(image_config.TOPO_THRESHOLD)),
                })

        if resume_manager is not None:
            resume_manager.complete(img_path, recovery_summary, profile_decision)

        print(f'Done for {img_id}.')

    known_timing_count = sum(
        "total_image_seconds" in row for row in recovery_summaries
    )
    time_txt = (
        f'Inference completed for {len(input_img_paths)} image(s) from {input_label} '
        f'with {total_inference_seconds} recorded seconds '
        f'(timing available for {known_timing_count}/{len(input_img_paths)} images; '
        f'validated marker skips: {resume_counts["validated_marker_skip_count"]}, '
        f'legacy adopted: {resume_counts["legacy_adopted_count"]}, '
        f'inferred: {resume_counts["inferred_count"]}).'
    )
    print(time_txt)
    with open(os.path.join(output_dir, 'inference_time.txt'), 'w', encoding='utf-8') as f:
        f.write(time_txt)
    total_fields = (
        'strong_edge_count', 'weak_candidate_count', 'weak_recovered_edge_count',
        'surface_supported_recovery_count', 'rejected_weak_candidate_count',
        'bootstrap_candidate_count', 'bootstrap_accepted_candidate_count',
        'bootstrap_recovered_edge_count',
        'bootstrap_auto_count', 'bootstrap_review_count', 'bootstrap_rejected_count',
        'relative_candidate_count', 'relative_accepted_candidate_count',
        'relative_recovered_edge_count', 'relative_auto_count',
        'relative_review_count', 'relative_rejected_count',
        'relative_compute_call_count', 'native_graph_point_count',
        'relative_graph_point_count', 'toponet_graph_point_count',
        'toponet_candidate_edge_count', 'toponet_pred_edge_count',
    )
    timing_fields = (
        'mask_inference_seconds', 'native_graph_and_toponet_seconds',
        'relative_roadness_seconds', 'weak_postprocess_seconds',
        'weak_diagnosis_seconds', 'weak_relative_context_seconds',
        'weak_bootstrap_seconds', 'weak_endpoint_recovery_seconds',
        'endpoint_to_segment_recovery_seconds',
        'weak_connectivity_statistics_seconds',
        'total_image_seconds', 'relative_final_centerline_length',
    )
    recovery_report = {
        'tile_count': len(recovery_summaries),
        'relative_roadness_enabled': bool(config.get('RELATIVE_ROADNESS_ENABLED', False)),
        'relative_injected_into_toponet': bool(
            config.get('RELATIVE_ROADNESS_ENABLED', False)
            and config.get('RELATIVE_INJECT_INTO_TOPONET', False)
        ),
        'relative_centerline_method': (
            'regularized_skeleton'
            if config.get('RELATIVE_REGULARIZED_SKELETON_EXPERIMENTAL', False)
            else 'continuous_trace'
            if config.get('RELATIVE_CONTINUOUS_TRACING_EXPERIMENTAL', False)
            else 'ribbon'
        ),
        'regularized_skeleton_active': bool(
            config.get('RELATIVE_ROADNESS_ENABLED', False)
            and config.get('RELATIVE_REGULARIZED_SKELETON_EXPERIMENTAL', False)
        ),
        'continuous_tracing_active': bool(
            config.get('RELATIVE_ROADNESS_ENABLED', False)
            and not config.get('RELATIVE_REGULARIZED_SKELETON_EXPERIMENTAL', False)
            and config.get('RELATIVE_CONTINUOUS_TRACING_EXPERIMENTAL', False)
        ),
        'junction_collapse_active': bool(
            config.get('RELATIVE_ROADNESS_ENABLED', False)
            and config.get('RELATIVE_REGULARIZED_SKELETON_EXPERIMENTAL', False)
            and config.get('RELATIVE_JUNCTION_COLLAPSE_EXPERIMENTAL', False)
        ),
        'endpoint_segment_recovery_active': bool(
            config.get('WEAK_SEGMENT_RECOVERY_ENABLED', False)
        ),
        **{name: int(sum(row.get(name, 0) for row in recovery_summaries)) for name in total_fields},
        **{
            name: float(sum(float(row.get(name, 0.0)) for row in recovery_summaries))
            for name in timing_fields
        },
        'recovery_reason_counts': {
            reason: int(sum(
                row.get('recovery_reason_counts', {}).get(reason, 0)
                for row in recovery_summaries
            ))
            for reason in sorted({
                reason
                for row in recovery_summaries
                for reason in row.get('recovery_reason_counts', {})
            })
        },
        'weak_recovery_reject_reason_counts': {
            reason: int(sum(
                row.get('weak_recovery_reject_reason_counts', {}).get(reason, 0)
                for row in recovery_summaries
            ))
            for reason in sorted({
                reason
                for row in recovery_summaries
                for reason in row.get('weak_recovery_reject_reason_counts', {})
            })
        },
        'bootstrap_reject_reason_counts': {
            reason: int(sum(
                row.get('bootstrap_reject_reason_counts', {}).get(reason, 0)
                for row in recovery_summaries
            ))
            for reason in sorted({
                reason
                for row in recovery_summaries
                for reason in row.get('bootstrap_reject_reason_counts', {})
            })
        },
        'tiles': recovery_summaries,
        'image_resume': {
            'enabled': bool(resume_existing_images),
            **resume_counts,
        },
        'summary_field_coverage': {
            name: int(sum(name in row for row in recovery_summaries))
            for name in (*total_fields, *timing_fields)
        },
    }
    with open(os.path.join(output_dir, 'weak_recovery_summary.json'), 'w', encoding='utf-8') as file:
        json.dump(recovery_report, file, ensure_ascii=False, indent=2)
    profile_report = graph_extraction.summarize_profile_decisions(
        _config_profile(config),
        config.get("SCENE_DIAGNOSTIC_REFERENCE_PROFILE", "default"),
        profile_decisions,
    )
    with open(os.path.join(output_dir, 'profile_decisions.json'), 'w', encoding='utf-8') as file:
        json.dump(profile_report, file, ensure_ascii=False, indent=2)
    return total_inference_seconds


def _config_profile(config):
    return config.get("ROAD_THRESHOLD_PROFILE", "default")


def get_geotiff_gsd(path):
    if rasterio is None:
        return None
    try:
        with rasterio.open(str(path)) as ds:
            transform = ds.transform
            x_gsd = abs(float(transform.a))
            y_gsd = abs(float(transform.e))
            if x_gsd > 0 and y_gsd > 0:
                return (x_gsd + y_gsd) / 2.0
    except Exception:
        return None
    return None


def get_inference_gsd(path, config):
    model_gsd = args.model_gsd if args.model_gsd is not None else float(config.get('MODEL_GSD', 0.5))
    input_gsd = args.input_gsd
    if input_gsd is None:
        if bool(config.get('INFER_AUTO_GSD_FROM_GEOTIFF', True)):
            input_gsd = get_geotiff_gsd(path)
    if input_gsd is None:
        configured_gsd = config.get('INFER_INPUT_GSD', None)
        if input_gsd is None and configured_gsd is not None:
            input_gsd = float(configured_gsd)
    if input_gsd is None:
        input_gsd = model_gsd
    return float(input_gsd), float(model_gsd)


def rescale_image_to_model_gsd(img, path, config):
    input_gsd, model_gsd = get_inference_gsd(path, config)
    if not bool(config.get('INFER_RESCALE_TO_MODEL_GSD', True)):
        return img, 1.0, input_gsd, model_gsd
    if input_gsd <= 0 or model_gsd <= 0:
        return img, 1.0, input_gsd, model_gsd

    resize_factor = input_gsd / model_gsd
    if abs(resize_factor - 1.0) < 1e-3:
        return img, 1.0, input_gsd, model_gsd

    height, width = img.shape[:2]
    new_width = max(1, int(round(width * resize_factor)))
    new_height = max(1, int(round(height * resize_factor)))
    interpolation = cv2.INTER_CUBIC if resize_factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(img, (new_width, new_height), interpolation=interpolation), resize_factor, input_gsd, model_gsd


def get_full_coverage_positions(image_extent, patch_size, overlap):
    if image_extent <= patch_size:
        return [0]
    stride = max(1, patch_size - 2 * max(0, int(overlap)))
    positions = list(range(0, image_extent - patch_size + 1, stride))
    last_start = image_extent - patch_size
    if positions[-1] != last_start:
        positions.append(last_start)
    return positions


def get_full_coverage_patch_info(image_width, image_height, patch_size, overlap):
    patch_info = []
    x_positions = get_full_coverage_positions(image_width, patch_size, overlap)
    y_positions = get_full_coverage_positions(image_height, patch_size, overlap)
    for x in x_positions:
        for y in y_positions:
            patch_info.append((0, (x, y), (x + patch_size, y + patch_size)))
    return patch_info


def pad_image_to_min_size(img, patch_size):
    image_height, image_width = img.shape[:2]
    pad_bottom = max(0, patch_size - image_height)
    pad_right = max(0, patch_size - image_width)
    if pad_bottom == 0 and pad_right == 0:
        return img
    return cv2.copyMakeBorder(
        img,
        0,
        pad_bottom,
        0,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def filter_graph_to_image_bounds(pred_nodes, pred_edges, image_height, image_width, edge_scores=None):
    edge_scores = np.asarray(edge_scores if edge_scores is not None else np.ones(pred_edges.shape[0]), dtype=np.float32)
    if pred_nodes.shape[0] == 0:
        return pred_nodes, pred_edges, edge_scores
    keep_mask = (
        (pred_nodes[:, 0] >= 0)
        & (pred_nodes[:, 0] < image_height)
        & (pred_nodes[:, 1] >= 0)
        & (pred_nodes[:, 1] < image_width)
    )
    kept_indices = np.nonzero(keep_mask)[0]
    if kept_indices.shape[0] == pred_nodes.shape[0]:
        return pred_nodes, pred_edges, edge_scores

    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_indices.tolist())}
    filtered_nodes = pred_nodes[kept_indices]
    filtered_edges = []
    filtered_scores = []
    for (src_idx, tgt_idx), score in zip(pred_edges.tolist(), edge_scores.tolist()):
        if src_idx in index_map and tgt_idx in index_map:
            filtered_edges.append((index_map[src_idx], index_map[tgt_idx]))
            filtered_scores.append(score)
    filtered_edges = np.array(filtered_edges, dtype=np.int32).reshape(-1, 2) if filtered_edges else np.zeros((0, 2), dtype=np.int32)
    return filtered_nodes, filtered_edges, np.asarray(filtered_scores, dtype=np.float32)

def crop_img_patch(img, x0, y0, x1, y1):
    return img[y0:y1, x0:x1, :]

def get_batch_img_patches(img, batch_patch_info):
    patches = []
    for _, (x0, y0), (x1, y1) in batch_patch_info:
        patch = crop_img_patch(img, x0, y0, x1, y1)
        patches.append(torch.tensor(patch, dtype=torch.float32))
    batch = torch.stack(patches, 0).contiguous()
    return batch


def prepare_toponet_batch_mask(
    batch_mask,
    batch_patch_info,
    enhanced_road_probability,
    *,
    execution_profile,
):
    """Replace only the Fast TopoNet road channel with aligned enhanced patches."""
    if str(execution_profile).casefold() != "fast":
        return batch_mask
    if enhanced_road_probability is None:
        raise ValueError("Fast TopoNet requires enhanced road probability")
    probability = np.asarray(enhanced_road_probability, dtype=np.float32)
    road_patches = []
    for _, (x0, y0), (x1, y1) in batch_patch_info:
        road_patches.append(probability[y0:y1, x0:x1])
    enhanced_batch = torch.as_tensor(
        np.stack(road_patches, axis=0),
        dtype=batch_mask.dtype,
        device=batch_mask.device,
    )
    if enhanced_batch.shape != batch_mask[:, 1, :, :].shape:
        raise ValueError(
            "Fast enhanced TopoNet patch shape mismatch: "
            f"{tuple(enhanced_batch.shape)} != {tuple(batch_mask[:, 1, :, :].shape)}"
        )
    updated_mask = batch_mask.clone()
    updated_mask[:, 1, :, :] = enhanced_batch
    return updated_mask

def infer_one_img(net, img, config, *, diagnostic_shape=None):
    # TODO(congrui): centralize these configs
    image_height, image_width = img.shape[:2]
    batch_size = config.INFER_BATCH_SIZE
    # list of (i, (x_begin, y_begin), (x_end, y_end))
    all_patch_info = get_full_coverage_patch_info(
        image_width,
        image_height,
        config.PATCH_SIZE,
        int(config.get('SAMPLE_MARGIN', 0)),
    )
    patch_num = len(all_patch_info)
    batch_num = (
        patch_num // batch_size
        if patch_num % batch_size == 0
        else patch_num // batch_size + 1
    )
    # [IMG_H, IMG_W]
    fused_keypoint_mask = torch.zeros(img.shape[0:2], dtype=torch.float32).to(args.device, non_blocking=False)
    fused_road_mask = torch.zeros(img.shape[0:2], dtype=torch.float32).to(args.device, non_blocking=False)
    pixel_counter = torch.zeros(img.shape[0:2], dtype=torch.float32).to(args.device, non_blocking=False)
    # stores img embeddings for toponet
    # list of [B, D, h, w], len=batch_num
    img_features = list()
    img_mask = list()
    _synchronize_cuda_for_timing()
    mask_start_seconds = time.perf_counter()
    for batch_index in range(batch_num):
        offset = batch_index * batch_size
        batch_patch_info = all_patch_info[offset : offset + batch_size]
        # tensor [B, H, W, C]
        batch_img_patches = get_batch_img_patches(img, batch_patch_info)

        with torch.no_grad():
            batch_img_patches = batch_img_patches.to(args.device, non_blocking=False)
            # [B, H, W, 2]
            mask_scores, patch_img_features = net.infer_masks_and_img_features(batch_img_patches)
            img_features.append(patch_img_features)
            mask_scores11 = mask_scores.permute(0, 3, 1, 2)#(0,3,1,2)
            img_mask.append(mask_scores11)
        # Aggregate masks
        for patch_index, patch_info in enumerate(batch_patch_info):
            _, (x0, y0), (x1, y1) = patch_info
            keypoint_patch, road_patch = mask_scores[patch_index, :, :, 0], mask_scores[patch_index, :, :, 1]
            fused_keypoint_mask[y0:y1, x0:x1] += keypoint_patch
            fused_road_mask[y0:y1, x0:x1] += road_patch
            pixel_counter[y0:y1, x0:x1] += torch.ones(road_patch.shape[0:2], dtype=torch.float32, device=args.device)

    valid_pixels = pixel_counter > 0
    fused_keypoint_mask = torch.where(
        valid_pixels,
        fused_keypoint_mask / torch.clamp(pixel_counter, min=1.0),
        torch.zeros_like(fused_keypoint_mask),
    )
    fused_road_mask = torch.where(
        valid_pixels,
        fused_road_mask / torch.clamp(pixel_counter, min=1.0),
        torch.zeros_like(fused_road_mask),
    )
    fast_float_probability = (
        fused_road_mask.detach().cpu().numpy().astype(np.float32, copy=False)
        if args.execution_profile == "fast" else None
    )
    # range 0-1 -> 0-255
    fused_keypoint_mask = (fused_keypoint_mask * 255).to(torch.uint8).cpu().numpy()
    fused_road_mask = (fused_road_mask * 255).to(torch.uint8).cpu().numpy()
    _synchronize_cuda_for_timing()
    mask_inference_seconds = float(time.perf_counter() - mask_start_seconds)

    print(fused_road_mask.shape)
    diagnostic_probability = fused_road_mask
    if diagnostic_shape is not None:
        diagnostic_height, diagnostic_width = diagnostic_shape
        diagnostic_probability = fused_road_mask[:diagnostic_height, :diagnostic_width]
    profile_decision = graph_extraction.resolve_effective_road_profile(
        diagnostic_probability, config
    )
    config.ROAD_THRESHOLD_PROFILE = profile_decision["effective_profile"]
    relative_injected = bool(
        config.get("RELATIVE_ROADNESS_ENABLED", False)
        and config.get("RELATIVE_INJECT_INTO_TOPONET", False)
    )
    relative_context_unpadded = None
    relative_context = None
    relative_roadness_seconds = 0.0
    relative_compute_call_count = 0
    if relative_injected:
        relative_start_seconds = time.perf_counter()
        relative_context_unpadded = graph_extraction.compute_relative_roadness(
            diagnostic_probability,
            config,
            scene_state=profile_decision["scene_confidence_state"],
        )
        relative_roadness_seconds = float(
            time.perf_counter() - relative_start_seconds
        )
        relative_compute_call_count = 1
        relative_context = graph_extraction.embed_relative_roadness_context(
            relative_context_unpadded, fused_road_mask.shape
        )

    native_graph_start_seconds = time.perf_counter()
    native_graph_points = graph_extraction.extract_graph_points(
        fused_keypoint_mask,
        fused_road_mask,
        config,
        relative_context=None,
    )
    graph_points = native_graph_points
    fast_enhancement_context = None
    enhanced_probability = None
    if args.execution_profile == "fast":
        fast_graph_high_threshold, _fast_low_threshold, _fast_profile = (
            graph_extraction.resolve_road_thresholds(config)
        )
        enhanced_probability, enhancement_diagnostics = (
            build_fast_enhanced_road_probability(
                fast_float_probability,
                high_threshold=fast_graph_high_threshold,
            )
        )
        enhanced_road_mask = np.rint(enhanced_probability * 255.0).astype(np.uint8)
        graph_points = graph_extraction.extract_graph_points(
            fused_keypoint_mask,
            enhanced_road_mask,
            config,
            relative_context=None,
        )
        fast_enhancement_context = {
            "enhanced_road_mask": enhanced_road_mask,
            "diagnostics": enhancement_diagnostics,
        }
    elif relative_injected:
        graph_points = graph_extraction.extract_graph_points(
            fused_keypoint_mask,
            fused_road_mask,
            config,
            relative_context=relative_context,
        )
    performance_summary = {
        "mask_inference_seconds": mask_inference_seconds,
        "native_graph_and_toponet_seconds": 0.0,
        "relative_roadness_seconds": relative_roadness_seconds,
        "weak_postprocess_seconds": 0.0,
        "total_image_seconds": 0.0,
        "relative_injected_into_toponet": relative_injected,
        "relative_compute_call_count": relative_compute_call_count,
        "native_graph_point_count": int(native_graph_points.shape[0]),
        "raw_graph_point_count": int(native_graph_points.shape[0]),
        "fast_graph_point_count": int(graph_points.shape[0]),
        "relative_graph_point_count": 0,
        "toponet_graph_point_count": int(graph_points.shape[0]),
        "toponet_candidate_edge_count": 0,
        "toponet_pred_edge_count": 0,
        "toponet_final_edge_count": 0,
        "final_centerline_length_px": 0.0,
        "relative_final_centerline_length": 0.0,
    }
    if fast_enhancement_context is not None:
        performance_summary.update(fast_enhancement_context["diagnostics"])
    profile_decision.update({
        "relative_injected_into_toponet": relative_injected,
        "native_graph_point_count": performance_summary["native_graph_point_count"],
        "toponet_graph_point_count": performance_summary["toponet_graph_point_count"],
    })
    if relative_context_unpadded is not None:
        profile_decision.update(relative_context_unpadded.get("diagnostics", {}))
    if graph_points.shape[0] == 0:
        print(1)
        print(graph_points)
        empty_edges = np.zeros((0, 2), dtype=np.int32)
        empty_scores = np.zeros((0,), dtype=np.float32)
        performance_summary["native_graph_and_toponet_seconds"] = float(
            time.perf_counter() - native_graph_start_seconds
        )
        return (
            graph_points, empty_edges, empty_scores, empty_edges, empty_scores,
            fused_keypoint_mask, fused_road_mask, profile_decision,
            fast_enhancement_context if args.execution_profile == "fast" else relative_context_unpadded,
            performance_summary,
        )
    # for box query
    graph_rtree = rtree.index.Index()
    for i, v in enumerate(graph_points):
        x, y = v
        # hack to insert single points
        graph_rtree.insert(i, (x, y, x, y))
    ## Pass 2: infer toponet to predict topology of points from stored img features
    edge_scores = defaultdict(float)
    edge_counts = defaultdict(float)
    for batch_index in range(batch_num):
        offset = batch_index * batch_size
        batch_patch_info = all_patch_info[offset : offset + batch_size]

        topo_data = {
            'points': [],
            'pairs': [],
            'valid': [],
        }
        idx_maps = []
        # prepares pairs queries
        for patch_info in batch_patch_info:
            _, (x0, y0), (x1, y1) = patch_info
            patch_point_indices = list(graph_rtree.intersection((x0, y0, x1, y1)))
            idx_patch2all = {patch_idx : all_idx for patch_idx, all_idx in enumerate(patch_point_indices)}
            patch_point_num = len(patch_point_indices)
            # normalize into patch
            patch_points = graph_points[patch_point_indices, :] - np.array([[x0, y0]], dtype=graph_points.dtype)
            # for knn and circle query
            patch_kdtree = scipy.spatial.KDTree(patch_points)
            # k+1 because the nearest one is always self
            # idx is to the patch subgraph
            knn_d, knn_idx = patch_kdtree.query(patch_points, k=config.MAX_NEIGHBOR_QUERIES + 1, distance_upper_bound=config.NEIGHBOR_RADIUS)
            # [patch_point_num, n_nbr]
            knn_idx = knn_idx[:, 1:]  # removes self
            # [patch_point_num, n_nbr] idx is to the patch subgraph
            src_idx = np.tile(
                np.arange(patch_point_num)[:, np.newaxis],
                (1, config.MAX_NEIGHBOR_QUERIES)
            )
            valid = knn_idx < patch_point_num
            tgt_idx = np.where(valid, knn_idx, src_idx)
            # [patch_point_num, n_nbr, 2]
            pairs = np.stack([src_idx, tgt_idx], axis=-1)

            topo_data['points'].append(patch_points)
            topo_data['pairs'].append(pairs)
            topo_data['valid'].append(valid)
            idx_maps.append(idx_patch2all)
        # collate
        collated = {}
        for key, x_list in topo_data.items():
            length = max([x.shape[0] for x in x_list])
            collated[key] = np.stack([
                np.pad(x, [(0, length - x.shape[0])] + [(0, 0)] * (len(x.shape) - 1))
                for x in x_list
            ], axis=0)
        # skips this batch if there's no points
        if collated['points'].shape[1] == 0:
            continue
        # infer toponet
        # [B, D, h, w]
        batch_features = img_features[batch_index]
        batch_mask = prepare_toponet_batch_mask(
            img_mask[batch_index],
            batch_patch_info,
            enhanced_probability,
            execution_profile=args.execution_profile,
        )
        # [B, N_sample, N_pair, 2]
        batch_points = torch.tensor(collated['points'], device=args.device)
        batch_pairs = torch.tensor(collated['pairs'], device=args.device)
        batch_valid = torch.tensor(collated['valid'], device=args.device)
        with torch.no_grad():
            # [B, N_samples, N_pairs, 1]
            topo_scores = net.infer_toponet(batch_features, batch_points, batch_pairs, batch_valid,batch_mask) 
        # all-invalid (padded, no neighbors) queries returns nan scores
        # [B, N_samples, N_pairs]
        topo_scores = torch.where(torch.isnan(topo_scores), -100.0, topo_scores).squeeze(-1).cpu().numpy()
        # aggregate edge scores
        batch_size, n_samples, n_pairs = topo_scores.shape
        for bi in range(batch_size):
            for si in range(n_samples):
                for pi in range(n_pairs):
                    if not collated['valid'][bi, si, pi]:
                        continue
                    # idx to the full graph
                    src_idx_patch, tgt_idx_patch = collated['pairs'][bi, si, pi, :]
                    src_idx_all, tgt_idx_all = idx_maps[bi][src_idx_patch], idx_maps[bi][tgt_idx_patch]
                    edge_score = topo_scores[bi, si, pi]
                    assert 0.0 <= edge_score <= 1.0
                    edge_scores[(src_idx_all, tgt_idx_all)] += edge_score
                    edge_counts[(src_idx_all, tgt_idx_all)] += 1.0
    # avg edge scores and filter
    pred_edges = []
    pred_edge_scores = []
    candidate_edges = []
    candidate_edge_scores = []
    candidate_threshold = float(config.get('TOPO_CANDIDATE_THRESHOLD', 0.20))
    for edge, score_sum in edge_scores.items():
        score = score_sum / edge_counts[edge] 
        if score > candidate_threshold:
            candidate_edges.append(edge)
            candidate_edge_scores.append(score)
        if score > config.TOPO_THRESHOLD:
            pred_edges.append(edge)
            pred_edge_scores.append(score)
    pred_edges = np.array(pred_edges).reshape(-1, 2)
    pred_edge_scores = np.asarray(pred_edge_scores, dtype=np.float32)
    candidate_edges = np.asarray(candidate_edges, dtype=np.int32).reshape(-1, 2)
    candidate_edge_scores = np.asarray(candidate_edge_scores, dtype=np.float32)
    pred_nodes = graph_points[:, ::-1]  # to rc
    _synchronize_cuda_for_timing()
    performance_summary["native_graph_and_toponet_seconds"] = float(
        time.perf_counter() - native_graph_start_seconds
    )
    performance_summary["toponet_candidate_edge_count"] = int(candidate_edges.shape[0])
    performance_summary["toponet_pred_edge_count"] = int(pred_edges.shape[0])
    performance_summary["toponet_final_edge_count"] = int(pred_edges.shape[0])
    unique_pred_edges = {
        tuple(sorted((int(src), int(dst))))
        for src, dst in pred_edges.tolist() if int(src) != int(dst)
    }
    performance_summary["final_centerline_length_px"] = float(sum(
        np.linalg.norm(pred_nodes[src] - pred_nodes[dst])
        for src, dst in unique_pred_edges
    ))

    return (
        pred_nodes,
        pred_edges,
        pred_edge_scores,
        candidate_edges,
        candidate_edge_scores,
        fused_keypoint_mask,
        fused_road_mask,
        profile_decision,
        fast_enhancement_context if args.execution_profile == "fast" else relative_context_unpadded,
        performance_summary,
    )

if __name__ == "__main__":
    config_path = resolve_repo_path(args.config).resolve()
    config = load_config(config_path)
    config.INFER_RESCALE_TO_MODEL_GSD = args.rescale_to_model_gsd == "on"
    if args.junction_node_mode:
        config.JUNCTION_NODE_MODE = args.junction_node_mode
    # Builds eval model    
    resolved_device_name, device = resolve_torch_device(args.device)
    args.device = resolved_device_name
    if resolved_device_name == "cuda":
        # Good when model architecture/input shape are fixed.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    net = SAMRoadplus(config)

    # load checkpoint
    checkpoint_path = resolve_repo_path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print(f'##### Loading Trained CKPT {checkpoint_path} #####')
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.eval()
    net.to(device)

    output_root = resolve_repo_path(args.output_root)
    output_dir_prefix = str(output_root / 'offline_infer')
    inference_sources = gather_inference_inputs(args.input_dir, args.input_txt_dir)

    if args.output_dir:
        specified_output_dir = Path(args.output_dir)
        if not specified_output_dir.is_absolute():
            specified_output_dir = output_root / specified_output_dir
        base_output_dir = create_output_dir_and_save_config(output_dir_prefix, config, specified_dir=specified_output_dir)
    else:
        base_output_dir = create_output_dir_and_save_config(output_dir_prefix, config)

    requested_profile = str(config.get('ROAD_THRESHOLD_PROFILE', 'default'))
    regularized_skeleton_active = bool(
        config.get('RELATIVE_ROADNESS_ENABLED', False)
        and config.get('RELATIVE_REGULARIZED_SKELETON_EXPERIMENTAL', False)
    )
    continuous_tracing_active = bool(
        config.get('RELATIVE_ROADNESS_ENABLED', False)
        and not regularized_skeleton_active
        and config.get('RELATIVE_CONTINUOUS_TRACING_EXPERIMENTAL', False)
    )
    identity_parameters = {
        'device': resolved_device_name,
        'rescale_to_model_gsd': args.rescale_to_model_gsd,
        'input_gsd': args.input_gsd,
        'model_gsd': args.model_gsd,
        'max_image_megapixels': args.max_image_megapixels,
        'execution_profile': args.execution_profile,
        'junction_node_mode': str(config.get('JUNCTION_NODE_MODE', '')),
        'topology_threshold': float(config.TOPO_THRESHOLD),
        'topology_candidate_threshold': float(config.get('TOPO_CANDIDATE_THRESHOLD', 0.20)),
        'requested_road_threshold_profile': requested_profile,
        'diagnostic_reference_profile': str(config.get('SCENE_DIAGNOSTIC_REFERENCE_PROFILE', 'default')),
        'weak_recovery_enabled': bool(config.get('WEAK_RECOVERY_ENABLED', True)),
        'weak_bootstrap_enabled': bool(config.get('WEAK_BOOTSTRAP_ENABLED', True)),
        'weak_bootstrap_only_if_low_confidence': bool(config.get('WEAK_BOOTSTRAP_ONLY_IF_LOW_CONFIDENCE', True)),
        'relative_roadness_enabled': bool(config.get('RELATIVE_ROADNESS_ENABLED', False)),
        'relative_injected_into_toponet': bool(
            config.get('RELATIVE_ROADNESS_ENABLED', False)
            and config.get('RELATIVE_INJECT_INTO_TOPONET', False)
        ),
    }
    resume_identity = build_batch_identity(checkpoint_path, config_path, identity_parameters)
    metadata_path = Path(base_output_dir) / 'inference_metadata.json'
    legacy_metadata = None
    if args.resume_existing_images and metadata_path.is_file():
        try:
            value = json.loads(metadata_path.read_text(encoding='utf-8'))
            legacy_metadata = value if isinstance(value, dict) else None
        except (OSError, UnicodeError, json.JSONDecodeError):
            legacy_metadata = None
    if isinstance(legacy_metadata, dict):
        saved_config = Path(base_output_dir) / 'config.yaml'
        if saved_config.is_file():
            try:
                legacy_metadata['_saved_config_identity'] = effective_config_identity(
                    saved_config, inspect_resources=False,
                )
            except (OSError, UnicodeError, ValueError):
                pass
    metadata = {
        'checkpoint': str(checkpoint_path),
        'config': str(config_path),
        'device': resolved_device_name,
        'execution_profile': args.execution_profile,
        'topology_threshold': identity_parameters['topology_threshold'],
        'topology_candidate_threshold': identity_parameters['topology_candidate_threshold'],
        'requested_road_threshold_profile': requested_profile,
        'diagnostic_reference_profile': identity_parameters['diagnostic_reference_profile'],
        'profile_selection_mode': 'automatic' if requested_profile == 'auto' else 'manual',
        'weak_recovery_enabled': identity_parameters['weak_recovery_enabled'],
        'weak_bootstrap_enabled': identity_parameters['weak_bootstrap_enabled'],
        'weak_bootstrap_only_if_low_confidence': identity_parameters['weak_bootstrap_only_if_low_confidence'],
        'relative_roadness_enabled': identity_parameters['relative_roadness_enabled'],
        'relative_injected_into_toponet': identity_parameters['relative_injected_into_toponet'],
        'relative_centerline_method': (
            'regularized_skeleton' if regularized_skeleton_active
            else 'continuous_trace' if continuous_tracing_active
            else 'ribbon'
        ),
        'regularized_skeleton_active': regularized_skeleton_active,
        'continuous_tracing_active': continuous_tracing_active,
        'junction_collapse_active': bool(
            regularized_skeleton_active
            and config.get('RELATIVE_JUNCTION_COLLAPSE_EXPERIMENTAL', False)
        ),
        'endpoint_segment_recovery_active': bool(config.get('WEAK_SEGMENT_RECOVERY_ENABLED', False)),
        'branch_aware_road_nms': True,
        'resume_identity': resume_identity,
    }
    for source_name, source_path, input_img_paths in inference_sources:
        if not input_img_paths:
            raise RuntimeError(f"No images found in inference source: {source_path}")
        if args.input_txt_dir:
            output_dir = os.path.join(base_output_dir, source_name)
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = base_output_dir

        run_inference_on_images(
            net, config, input_img_paths, output_dir, str(source_path),
            resume_existing_images=args.resume_existing_images,
            resume_identity=resume_identity,
            legacy_metadata=legacy_metadata,
            pipeline_state=(args.pipeline_state or None),
        )

    metadata_temporary = metadata_path.with_name(
        f'.{metadata_path.name}.{os.getpid()}.tmp'
    )
    try:
        metadata_temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8',
        )
        os.replace(metadata_temporary, metadata_path)
    finally:
        metadata_temporary.unlink(missing_ok=True)

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
from argparse import ArgumentParser
from pathlib import Path
from package_paths import INFER_RUNS_ROOT, PROJECT_ROOT, resolve_path
from input_catalog import read_path_list

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
    listing = read_path_list(txt_path, search_roots=(PROJECT_ROOT, Path.cwd()))
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


def run_inference_on_images(net, config, input_img_paths, output_dir, input_label):
    total_inference_seconds = 0.0
    recovery_summaries = []
    print(f'Found {len(input_img_paths)} image(s) under {input_label}.')
    print(f'Inference patch size: {config.PATCH_SIZE}x{config.PATCH_SIZE}')
    print(f'Inference device: {resolved_device_name}')

    for img_path in input_img_paths:
        img_path = Path(img_path)
        img_id = img_path.stem
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

        start_seconds = time.time()
        (
            pred_nodes,
            pred_edges,
            edge_confidences,
            candidate_edges,
            candidate_confidences,
            itsc_mask,
            road_mask,
        ) = infer_one_img(net, padded_img, config)
        total_inference_seconds += (time.time() - start_seconds)

        itsc_mask = itsc_mask[:infer_height, :infer_width]
        road_mask = road_mask[:infer_height, :infer_width]
        candidate_nodes, candidate_edges, candidate_confidences = filter_graph_to_image_bounds(
            pred_nodes, candidate_edges, infer_height, infer_width, candidate_confidences
        )
        pred_nodes, pred_edges, edge_confidences = filter_graph_to_image_bounds(
            pred_nodes, pred_edges, infer_height, infer_width, edge_confidences
        )
        if resize_factor != 1.0:
            itsc_mask = cv2.resize(itsc_mask, (original_width, original_height), interpolation=cv2.INTER_AREA)
            road_mask = cv2.resize(road_mask, (original_width, original_height), interpolation=cv2.INTER_AREA)
            pred_nodes = pred_nodes.astype(np.float32) / float(resize_factor)
            candidate_nodes = candidate_nodes.astype(np.float32) / float(resize_factor)
            candidate_nodes, candidate_edges, candidate_confidences = filter_graph_to_image_bounds(
                candidate_nodes, candidate_edges, original_height, original_width, candidate_confidences
            )
            pred_nodes, pred_edges, edge_confidences = filter_graph_to_image_bounds(
                pred_nodes, pred_edges, original_height, original_width, edge_confidences
            )

        pred_nodes, pred_edges, edge_metadata, recovery_summary = graph_extraction.postprocess_weak_road_network(
            pred_nodes,
            pred_edges,
            road_mask,
            config,
            edge_scores=edge_confidences,
            distance_scale=1.0 / max(float(resize_factor), 1e-6),
        )
        edge_confidences = np.asarray(
            [row["topology_probability"] for row in edge_metadata], dtype=np.float32
        )
        recovery_summary = {"tile": img_id, **recovery_summary}
        recovery_summaries.append(recovery_summary)

        viz_img = np.copy(img)
        mask_save_dir = os.path.join(output_dir, 'mask')
        os.makedirs(mask_save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(mask_save_dir, f'{img_id}_road.png'), road_mask)
        cv2.imwrite(os.path.join(mask_save_dir, f'{img_id}_itsc.png'), itsc_mask)

        viz_save_dir = os.path.join(output_dir, 'viz')
        os.makedirs(viz_save_dir, exist_ok=True)
        norm_scale = np.array([[viz_img.shape[0], viz_img.shape[1]]], dtype=np.float32)
        viz_img = triage.visualize_image_and_graph(viz_img, pred_nodes / norm_scale, pred_edges, viz_img.shape[0])
        cv2.imwrite(os.path.join(viz_save_dir, f'{img_id}.png'), viz_img)

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
                })
        with open(os.path.join(graph_save_dir, f'{img_id}_weak_recovery.json'), 'w', encoding='utf-8') as file:
            json.dump(recovery_summary, file, ensure_ascii=False, indent=2)
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
                    'selected': int(float(score) > float(config.TOPO_THRESHOLD)),
                })

        print(f'Done for {img_id}.')

    time_txt = (
        f'Inference completed for {len(input_img_paths)} image(s) from {input_label} '
        f'in {total_inference_seconds} seconds.'
    )
    print(time_txt)
    with open(os.path.join(output_dir, 'inference_time.txt'), 'w', encoding='utf-8') as f:
        f.write(time_txt)
    total_fields = (
        'strong_edge_count', 'weak_candidate_count', 'weak_recovered_edge_count',
        'surface_supported_recovery_count', 'rejected_weak_candidate_count',
        'bootstrap_candidate_count', 'bootstrap_recovered_edge_count',
        'bootstrap_auto_count', 'bootstrap_review_count', 'bootstrap_rejected_count',
    )
    recovery_report = {
        'tile_count': len(recovery_summaries),
        **{name: int(sum(row.get(name, 0) for row in recovery_summaries)) for name in total_fields},
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
        'tiles': recovery_summaries,
    }
    with open(os.path.join(output_dir, 'weak_recovery_summary.json'), 'w', encoding='utf-8') as file:
        json.dump(recovery_report, file, ensure_ascii=False, indent=2)
    return total_inference_seconds


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

def infer_one_img(net, img, config):
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
    # range 0-1 -> 0-255
    fused_keypoint_mask = (fused_keypoint_mask * 255).to(torch.uint8).cpu().numpy()
    fused_road_mask = (fused_road_mask * 255).to(torch.uint8).cpu().numpy()

    print(fused_road_mask.shape)
    graph_points = graph_extraction.extract_graph_points(fused_keypoint_mask, fused_road_mask, config)
    if graph_points.shape[0] == 0:
        print(1)
        print(graph_points)
        empty_edges = np.zeros((0, 2), dtype=np.int32)
        empty_scores = np.zeros((0,), dtype=np.float32)
        return graph_points, empty_edges, empty_scores, empty_edges, empty_scores, fused_keypoint_mask, fused_road_mask
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
        batch_mask = img_mask[batch_index]
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

    return (
        pred_nodes,
        pred_edges,
        pred_edge_scores,
        candidate_edges,
        candidate_edge_scores,
        fused_keypoint_mask,
        fused_road_mask,
    )

if __name__ == "__main__":
    config = load_config(args.config)
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
    checkpoint_path = resolve_repo_path(args.checkpoint)
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

    road_high_threshold, road_low_threshold, road_threshold_profile = graph_extraction.resolve_road_thresholds(config)
    with open(os.path.join(base_output_dir, 'inference_metadata.json'), 'w', encoding='utf-8') as file:
        json.dump(
            {
                'checkpoint': str(checkpoint_path),
                'config': str(resolve_repo_path(args.config)),
                'device': resolved_device_name,
                'topology_threshold': float(config.TOPO_THRESHOLD),
                'topology_candidate_threshold': float(config.get('TOPO_CANDIDATE_THRESHOLD', 0.20)),
                'road_threshold_profile': road_threshold_profile,
                'road_high_threshold': road_high_threshold,
                'road_low_threshold': road_low_threshold,
                'weak_recovery_enabled': bool(config.get('WEAK_RECOVERY_ENABLED', True)),
                'weak_bootstrap_enabled': bool(config.get('WEAK_BOOTSTRAP_ENABLED', True)),
                'weak_bootstrap_only_if_low_confidence': bool(
                    config.get('WEAK_BOOTSTRAP_ONLY_IF_LOW_CONFIDENCE', True)
                ),
                'branch_aware_road_nms': True,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    for source_name, source_path, input_img_paths in inference_sources:
        if not input_img_paths:
            raise RuntimeError(f"No images found in inference source: {source_path}")
        if args.input_txt_dir:
            output_dir = os.path.join(base_output_dir, source_name)
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = base_output_dir

        run_inference_on_images(net, config, input_img_paths, output_dir, str(source_path))

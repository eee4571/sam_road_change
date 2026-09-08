import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import math
import graph_utils
import rtree
import scipy
import pickle
import os
import addict
import json
from pathlib import Path
from package_paths import PROJECT_ROOT, existing_path, resolve_path

try:
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None


def resolve_repo_path(path):
    return resolve_path(path, PROJECT_ROOT)

def _to_uint8(image):
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        if info.max <= 255 and info.min >= 0:
            return image.astype(np.uint8)
        image = image.astype(np.float32)
        denom = max(1.0, float(info.max - info.min))
        image = (image - float(info.min)) / denom * 255.0
        return np.clip(image, 0, 255).astype(np.uint8)
    image = np.nan_to_num(image, nan=0.0)
    min_val = float(np.nanmin(image))
    max_val = float(np.nanmax(image))
    if max_val <= 1.0 and min_val >= 0.0:
        image = image * 255.0
    else:
        denom = max(1e-6, max_val - min_val)
        image = (image - min_val) / denom * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)

def read_rgb_img(path):
    bgr = cv2.imread(str(path))
    if bgr is not None:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if rasterio is not None:
        with rasterio.open(str(path)) as ds:
            band_count = max(1, min(3, ds.count))
            bands = ds.read(list(range(1, band_count + 1)))
            if bands.shape[0] == 1:
                bands = np.repeat(bands, 3, axis=0)
            elif bands.shape[0] == 2:
                bands = np.concatenate([bands, bands[-1:]], axis=0)
            else:
                bands = bands[:3]
            rgb = np.transpose(bands, (1, 2, 0))
            return _to_uint8(rgb)

    raise FileNotFoundError(f"Could not read RGB image: {path}")


def apply_radiometric_aug(rgb, config):
    img = rgb.astype(np.float32)

    if np.random.rand() < float(config.get('AUG_BRIGHTNESS_PROB', 0.8)):
        lo, hi = config.get('AUG_BRIGHTNESS_RANGE', [0.85, 1.15])
        img *= np.random.uniform(float(lo), float(hi))

    if np.random.rand() < float(config.get('AUG_CONTRAST_PROB', 0.8)):
        lo, hi = config.get('AUG_CONTRAST_RANGE', [0.85, 1.2])
        factor = np.random.uniform(float(lo), float(hi))
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = (img - mean) * factor + mean

    if np.random.rand() < float(config.get('AUG_GAMMA_PROB', 0.5)):
        lo, hi = config.get('AUG_GAMMA_RANGE', [0.85, 1.2])
        gamma = np.random.uniform(float(lo), float(hi))
        img = 255.0 * np.power(np.clip(img, 0, 255) / 255.0, gamma)

    if np.random.rand() < float(config.get('AUG_CHANNEL_SCALE_PROB', 0.5)):
        lo, hi = config.get('AUG_CHANNEL_SCALE_RANGE', [0.9, 1.1])
        channel_scale = np.random.uniform(float(lo), float(hi), size=(1, 1, 3))
        img *= channel_scale

    if np.random.rand() < float(config.get('AUG_NOISE_PROB', 0.3)):
        lo, hi = config.get('AUG_NOISE_SIGMA_RANGE', [2.0, 6.0])
        sigma = np.random.uniform(float(lo), float(hi))
        img += np.random.normal(0.0, sigma, img.shape)

    img = np.clip(img, 0, 255).astype(np.uint8)

    if np.random.rand() < float(config.get('AUG_BLUR_PROB', 0.15)):
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def apply_sensor_style_aug(rgb, config):
    img = rgb.astype(np.float32)
    if np.random.rand() < float(config.get('AUG_SENSOR_COLOR_PROB', 0.8)):
        channel_lo, channel_hi = config.get('AUG_SENSOR_CHANNEL_RANGE', [0.75, 1.25])
        channel_scale = np.random.uniform(float(channel_lo), float(channel_hi), size=(1, 1, 3))
        channel_bias = np.random.uniform(-12.0, 12.0, size=(1, 1, 3))
        img = img * channel_scale + channel_bias

    img = np.clip(img, 0, 255).astype(np.uint8)
    if np.random.rand() < float(config.get('AUG_SENSOR_SATURATION_PROB', 0.7)):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat_lo, sat_hi = config.get('AUG_SENSOR_SATURATION_RANGE', [0.55, 1.35])
        hsv[..., 1] *= np.random.uniform(float(sat_lo), float(sat_hi))
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

    if np.random.rand() < float(config.get('AUG_SENSOR_BLUR_PROB', 0.25)):
        sigma = np.random.uniform(0.3, 1.2)
        img = cv2.GaussianBlur(img, (3, 3), sigma)
    if np.random.rand() < float(config.get('AUG_SENSOR_NOISE_PROB', 0.35)):
        sigma = np.random.uniform(1.0, 8.0)
        img = np.clip(img.astype(np.float32) + np.random.normal(0.0, sigma, img.shape), 0, 255).astype(np.uint8)
    return img


def apply_resolution_aug(rgb, config):
    if np.random.rand() >= float(config.get('AUG_RESOLUTION_PROB', 0.6)):
        return rgb

    base_gsd = float(config.get('AUG_RESOLUTION_BASE_GSD', 0.5))
    lo, hi = config.get('AUG_RESOLUTION_TARGET_GSD_RANGE', [0.5, 1.0])
    target_gsd = np.random.uniform(float(lo), float(hi))
    if base_gsd <= 0 or target_gsd <= 0:
        return rgb

    scale = min(1.0, base_gsd / target_gsd)
    height, width = rgb.shape[:2]
    small_width = max(1, int(round(width * scale)))
    small_height = max(1, int(round(height * scale)))
    if small_width == width and small_height == height:
        return rgb

    small = cv2.resize(rgb, (small_width, small_height), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def cityscale_data_partition(total=180):
     # dataset partition
    indrange_train = []
    indrange_test = []
    indrange_validation = []

    for x in range(total):
        if x % 10 < 8 :
            indrange_train.append(x)

        if x % 10 == 9:
            indrange_test.append(x)

        if x % 20 == 18:
            indrange_validation.append(x)

        if x % 20 == 8:
            indrange_test.append(x)
    return indrange_train, indrange_validation, indrange_test


def cityscale_data_partition_from_count(total):
    if total <= 0:
        return [], [], []
    indrange_train = []
    indrange_test = []
    indrange_validation = []
    for x in range(total):
        if x % 10 < 8:
            indrange_train.append(x)
        if x % 20 == 18:
            indrange_validation.append(x)
        if x % 10 == 9 or x % 20 == 8:
            indrange_test.append(x)
    return indrange_train, indrange_validation, indrange_test


def cityscale_data_partition_from_ids(tile_ids):
    tile_ids = sorted(int(x) for x in tile_ids)
    if not tile_ids:
        return [], [], []
    train = []
    val = []
    test = []
    for tile_id in tile_ids:
        if tile_id % 10 < 8:
            train.append(tile_id)
        if tile_id % 20 == 18:
            val.append(tile_id)
        if tile_id % 10 == 9 or tile_id % 20 == 8:
            test.append(tile_id)
    return train, val, test


def normalize_tile_ids(tile_ids):
    if tile_ids is None:
        return None
    return [int(x) for x in tile_ids]


def get_cityscale_split_from_config(config):
    train_ids = normalize_tile_ids(config.get('TRAIN_TILE_IDS', None))
    val_ids = normalize_tile_ids(config.get('VAL_TILE_IDS', None))
    test_ids = normalize_tile_ids(config.get('TEST_TILE_IDS', None))
    if train_ids is not None or val_ids is not None or test_ids is not None:
        return train_ids or [], val_ids or [], test_ids or [], True

    active_tile_ids = normalize_tile_ids(config.get('ACTIVE_TILE_IDS', None))
    if active_tile_ids:
        return (*cityscale_data_partition_from_ids(active_tile_ids), False)

    return (*cityscale_data_partition_from_count(int(config.get('CITYSCALE_TILE_NUM', 180))), False)

def globalscale_data_partition():
    # dataset partition
    indrange_train = []
    indrange_test = []
    indrange_test_out_domain = []
    indrange_validation = []
    #0-2374 train
    #2375-2713 val
    #2714-3337 indomain
    for x in range(2375):
        indrange_train.append(x)
    
    for x in range(2375,2714):
        indrange_validation.append(x)

    for x in range(2714,3338):
        indrange_test.append(x)
    
    for x in range(130):
        indrange_test_out_domain.append(x)
    return indrange_train, indrange_validation, indrange_test,indrange_test_out_domain

def spacenet_data_partition():
    # dataset partition
    with open(resolve_repo_path('spacenet/data_split.json'), 'r', encoding='utf-8-sig') as jf:
        data_list = json.load(jf)
    train_list = data_list['train']
    val_list = data_list['validation']
    test_list = data_list['test']
    return train_list, val_list, test_list

def get_axis_patch_range(image_extent, sample_margin, patch_size):
    if image_extent < patch_size:
        raise ValueError(f'Image extent {image_extent} is smaller than patch size {patch_size}.')

    max_start = image_extent - patch_size
    if max_start == 0:
        return 0, 0

    sample_margin = max(0, int(sample_margin))
    constrained_min = min(sample_margin, max_start)
    constrained_max = max_start - sample_margin
    if constrained_max < constrained_min:
        return 0, max_start
    return constrained_min, constrained_max


def get_eval_patches_per_axis(image_extent, sample_margin, patch_size):
    sample_min, sample_max = get_axis_patch_range(image_extent, sample_margin, patch_size)
    return max(1, math.ceil((sample_max - sample_min) / patch_size) + 1)


def get_patch_info_one_img(image_index, image_width, image_height, sample_margin, patch_size, patches_per_width, patches_per_height=None):
    patch_info = []
    patches_per_height = patches_per_width if patches_per_height is None else patches_per_height
    x_min, x_max = get_axis_patch_range(image_width, sample_margin, patch_size)
    y_min, y_max = get_axis_patch_range(image_height, sample_margin, patch_size)
    eval_samples_x = np.linspace(start=x_min, stop=x_max, num=max(1, int(patches_per_width)))
    eval_samples_y = np.linspace(start=y_min, stop=y_max, num=max(1, int(patches_per_height)))
    eval_samples_x = sorted({round(x) for x in eval_samples_x})
    eval_samples_y = sorted({round(y) for y in eval_samples_y})
    for x in eval_samples_x:
        for y in eval_samples_y:
            patch_info.append(
                (image_index, (x, y), (x + patch_size, y + patch_size))
            )
    return patch_info

class GraphLabelGenerator():
    def __init__(self, config, full_graph, coord_transform):
        self.config = config
        # full_graph: sat2graph format
        # coord_transform: lambda, [N, 2] array -> [N, 2] array
        # convert to igraph for high performance
        self.full_graph_origin = graph_utils.igraph_from_adj_dict(full_graph, coord_transform)
        # find crossover points, we'll avoid predicting these as keypoints
        self.crossover_points = graph_utils.find_crossover_points(self.full_graph_origin)
        # subdivide version
        # TODO: check proper resolution
        self.subdivide_resolution = 4
        self.full_graph_subdivide = graph_utils.subdivide_graph(self.full_graph_origin, self.subdivide_resolution)
        # np array, maybe faster
        self.subdivide_points = np.array(self.full_graph_subdivide.vs['point'])
        self._build_spatial_indices()

        # pre-exclude points near crossover points
        crossover_exclude_radius = 4
        exclude_indices = set()
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(p, crossover_exclude_radius)
            exclude_indices.update(nearby_indices)
        self.exclude_indices = exclude_indices

        # Find intersection points, these will always be kept in nms
        itsc_indices = set()
        point_num = len(self.full_graph_subdivide.vs)
        for i in range(point_num):
            if self.full_graph_subdivide.degree(i) != 2:
                itsc_indices.add(i)
        self.nms_score_override = np.zeros((point_num, ), dtype=np.float32)
        self.nms_score_override[np.array(list(itsc_indices))] = 2.0  # itsc points will always be kept

        # Points near crossover and intersections are interesting.
        # they will be more frequently sampled
        interesting_indices = set()
        interesting_radius = 32
        # near itsc
        for i in itsc_indices:
            p = self.subdivide_points[i]
            nearby_indices = self.graph_kdtree.query_ball_point(p, interesting_radius)
            interesting_indices.update(nearby_indices)
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(np.array(p), interesting_radius)
            interesting_indices.update(nearby_indices)
        self.sample_weights = np.full((point_num, ), 0.1, dtype=np.float32)
        self.sample_weights[list(interesting_indices)] = 0.9

    def _build_spatial_indices(self):
        # In-memory R-tree contents are not preserved when Windows DataLoader
        # workers spawn and pickle the dataset, so each process builds its own.
        self.graph_rtee = rtree.index.Index()
        for i, (x, y) in enumerate(self.subdivide_points):
            self.graph_rtee.insert(i, (x, y, x, y))
        self.graph_kdtree = scipy.spatial.KDTree(self.subdivide_points)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['graph_rtee'] = None
        state['graph_kdtree'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build_spatial_indices()
    
    def sample_patch(self, patch, rot_index = 0):
        (x0, y0), (x1, y1) = patch
        query_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        patch_indices_all = set(self.graph_rtee.intersection(query_box))
        patch_indices = patch_indices_all - self.exclude_indices
        # Use NMS to downsample, params shall resemble inference time
        patch_indices = np.array(list(patch_indices))
        if len(patch_indices) == 0:
            # print("==== Patch is empty ====")
            # this shall be rare, but if no points in side the patch, return null stuff
            sample_num = self.config.TOPO_SAMPLE_NUM
            max_nbr_queries = self.config.MAX_NEIGHBOR_QUERIES
            fake_points = np.array([[0.0, 0.0]], dtype=np.float32)
            fake_sample = ([[0, 0]] * max_nbr_queries, [False] * max_nbr_queries, [False] * max_nbr_queries)
            return fake_points, [fake_sample] * sample_num
        patch_points = self.subdivide_points[patch_indices, :]     
        # random scores to emulate different random configurations that all share a
        # similar spacing between sampled points
        # raise scores for intersction points so they are always kept
        nms_scores = np.random.uniform(low=0.9, high=1.0, size=patch_indices.shape[0])
        nms_score_override = self.nms_score_override[patch_indices]
        nms_scores = np.maximum(nms_scores, nms_score_override)
        nms_radius = self.config.ROAD_NMS_RADIUS    
        # kept_indces are into the patch_points array
        nmsed_points, kept_indices = graph_utils.nms_points(patch_points, nms_scores, radius=nms_radius, return_indices=True)
        # now this is into the subdivide graph
        nmsed_indices = patch_indices[kept_indices]
        nmsed_point_num = nmsed_points.shape[0]

        sample_num = self.config.TOPO_SAMPLE_NUM  # has to be greater than 1
        sample_weights = self.sample_weights[nmsed_indices]
        # indices into the nmsed points in the patch
        sample_indices_in_nmsed = np.random.choice(
            np.arange(start=0, stop=nmsed_points.shape[0], dtype=np.int32),
            size=sample_num, replace=True, p=sample_weights / np.sum(sample_weights))
        # indices into the subdivided graph
        sample_indices = nmsed_indices[sample_indices_in_nmsed]

        radius = self.config.NEIGHBOR_RADIUS
        max_nbr_queries = self.config.MAX_NEIGHBOR_QUERIES  # has to be greater than 1
        nmsed_kdtree = scipy.spatial.KDTree(nmsed_points)
        sampled_points = self.subdivide_points[sample_indices, :]
        # [n_sample, n_nbr]
        # k+1 because the nearest one is always self
        knn_d, knn_idx = nmsed_kdtree.query(sampled_points, k=max_nbr_queries + 1, distance_upper_bound=radius)

        samples = []
        for i in range(sample_num):
            source_node = sample_indices[i]
            valid_nbr_indices = knn_idx[i, knn_idx[i, :] < nmsed_point_num]
            valid_nbr_indices = valid_nbr_indices[1:] # the nearest one is self so remove
            target_nodes = [nmsed_indices[ni] for ni in valid_nbr_indices]  
            ### BFS to find immediate neighbors on graph
            reached_nodes = graph_utils.bfs_with_conditions(self.full_graph_subdivide, source_node, set(target_nodes), radius // self.subdivide_resolution)
            shall_connect = [t in reached_nodes for t in target_nodes]
            ###
            pairs = []
            valid = []
            source_nmsed_idx = sample_indices_in_nmsed[i]
            for target_nmsed_idx in valid_nbr_indices:
                pairs.append((source_nmsed_idx, target_nmsed_idx))
                valid.append(True)
            # zero-pad
            for i in range(len(pairs), max_nbr_queries):
                pairs.append((source_nmsed_idx, source_nmsed_idx))
                shall_connect.append(False)
                valid.append(False)
            samples.append((pairs, shall_connect, valid))
        # Transform points
        # [N, 2]
        nmsed_points -= np.array([x0, y0])[np.newaxis, :]
        # homo for rot
        # [N, 3]
        nmsed_points = np.concatenate([nmsed_points, np.ones((nmsed_point_num, 1), dtype=nmsed_points.dtype)], axis=1)
        trans = np.array([
            [1, 0, -0.5 * self.config.PATCH_SIZE],
            [0, 1, -0.5 * self.config.PATCH_SIZE],
            [0, 0, 1],
        ], dtype=np.float32)
        # ccw 90 deg in img (x, y)
        rot = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float32)
        nmsed_points = nmsed_points @ trans.T @ np.linalg.matrix_power(rot.T, rot_index) @ np.linalg.inv(trans.T)
        nmsed_points = nmsed_points[:, :2]
        return nmsed_points, samples
         
def graph_collate_fn(batch):
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        if key == 'graph_points':
            tensors = [item[key] for item in batch]
            max_point_num = max([x.shape[0] for x in tensors])
            padded = []
            for x in tensors:
                pad_num = max_point_num - x.shape[0]
                padded_x = torch.concat([x, torch.zeros(pad_num, 2)], dim=0)
                padded.append(padded_x)
            collated[key] = torch.stack(padded, dim=0)
        else:
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
    return collated

class SatMapDataset(Dataset):
    def __init__(self, config, is_train, dev_run=False):
        self.config = config
        assert self.config.DATASET in {'cityscale','globalscale', 'spacenet'}
        data_root = resolve_repo_path(self.config.get('DATA_ROOT', '.'))
        if self.config.DATASET == 'cityscale':
            self.IMAGE_SIZE = None
            self.SAMPLE_MARGIN = int(self.config.get('SAMPLE_MARGIN', 64))
            cityscale_root = existing_path(
                data_root / self.config.get('CITYSCALE_DIR', 'cityscale/20cities'),
                PROJECT_ROOT / 'custom_dataset' / '20cities',
                PROJECT_ROOT / 'cityscale' / '20cities',
            )
            processed_root = existing_path(
                data_root / self.config.get('PROCESSED_DIR', 'processed'),
                PROJECT_ROOT / 'custom_dataset' / 'processed',
                PROJECT_ROOT / 'processed',
            )
            rgb_pattern = str(cityscale_root / 'region_{}_sat.png')
            keypoint_mask_pattern = str(processed_root / 'keypoint_mask_{}.png')
            road_mask_pattern = str(processed_root / 'road_mask_{}.png')
            gt_graph_pattern = str(cityscale_root / 'region_{}_refine_gt_graph.p')

            metadata_path = cityscale_root.parent / 'metadata.json'
            metadata_tiles = {}
            if metadata_path.exists():
                with metadata_path.open('r', encoding='utf-8-sig') as file:
                    metadata = json.load(file)
                metadata_tiles = {
                    int(tile['tile_id']): tile
                    for tile in metadata.get('tiles', [])
                    if 'tile_id' in tile
                }

            train, val, test, explicit_split = get_cityscale_split_from_config(self.config)
        
            # coord-transform = (r, c) -> (x, y)
            # takes [N, 2] points
            coord_transform = lambda v : v[:, ::-1]

        elif self.config.DATASET == 'globalscale':
            self.IMAGE_SIZE = 2048
            self.SAMPLE_MARGIN = int(self.config.get('SAMPLE_MARGIN', 64))
            globalscale_root = data_root / self.config.get('GLOBALSCALE_DIR', 'globalscale')
            processed_root = data_root / self.config.get('PROCESSED_DIR', 'processed')
            rgb_pattern = str(globalscale_root / 'region_{}_sat.png')
            keypoint_mask_pattern = str(processed_root / 'keypoint_mask_{}.png')
            road_mask_pattern = str(processed_root / 'road_mask_{}.png')
            gt_graph_pattern = str(globalscale_root / 'region_{}_refine_gt_graph.p')

            train, val, test, test_out = globalscale_data_partition()
            explicit_split = False
            
            # coord-transform = (r, c) -> (x, y)
            # takes [N, 2] points
            coord_transform = lambda v : v[:, ::-1]

        elif self.config.DATASET == 'spacenet':
            self.IMAGE_SIZE = 400
            self.SAMPLE_MARGIN = int(self.config.get('SAMPLE_MARGIN', 0))
            spacenet_root = data_root / self.config.get('SPACENET_DIR', 'spacenet/RGB_1.0_meter')
            processed_root = data_root / self.config.get('PROCESSED_DIR', 'processed')
            rgb_pattern = str(spacenet_root / '{}__rgb.png')
            keypoint_mask_pattern = str(processed_root / 'keypoint_mask_{}.png')
            road_mask_pattern = str(processed_root / 'road_mask_{}.png')
            gt_graph_pattern = str(spacenet_root / '{}__gt_graph.p')
            
            train, val, test = spacenet_data_partition()
            explicit_split = False

            # coord-transform ??? -> (x, y)
            # takes [N, 2] points
            coord_transform = lambda v : np.stack([v[:, 1], 400 - v[:, 0]], axis=1)

        self.is_train = is_train

        train_split = train if explicit_split else train + val
        eval_split = val if explicit_split else test
        self.tile_indices = train_split if self.is_train else eval_split
        # Stores all imgs in memory.
        self.rgbs, self.keypoint_masks, self.road_masks  = [], [], []
        self.image_shapes = []
        # For graph label generation.
        self.graph_label_generators = []
        self.tile_metadata = []
        self.train_patch_count = 0

        ##### FAST DEBUG
        self.tile_ids = []
        tile_indices = self.tile_indices[:]
        if dev_run:
            tile_indices = tile_indices[:4]
        ##### FAST DEBUG
        for tile_idx in tile_indices:
            print(f'loading tile {tile_idx}')
            rgb_path = rgb_pattern.format(tile_idx)
            road_mask_path = road_mask_pattern.format(tile_idx)
            keypoint_mask_path = keypoint_mask_pattern.format(tile_idx)
            if not (Path(rgb_path).exists() and Path(road_mask_path).exists() and Path(keypoint_mask_path).exists() and Path(gt_graph_pattern.format(tile_idx)).exists()):
                print(f'===== skipped missing tile {tile_idx} =====')
                continue
            # graph label gen
            # gt graph: dict for adj list, for cityscale set keys are (r, c) nodes, values are list of (r, c) nodes
            # I don't know what coord system spacenet uses but we convert them all to (x, y)
            gt_graph_adj = pickle.load(open(gt_graph_pattern.format(tile_idx),'rb'))
            if len(gt_graph_adj) == 0:
                print(f'===== skipped empty tile {tile_idx} =====')
                continue
            rgb = read_rgb_img(rgb_path)
            self.rgbs.append(rgb)
            road_mask = cv2.imread(road_mask_path, cv2.IMREAD_GRAYSCALE)
            keypoint_mask = cv2.imread(keypoint_mask_path, cv2.IMREAD_GRAYSCALE)
            if road_mask is None:
                raise FileNotFoundError(f"Could not read road mask: {road_mask_path}")
            if keypoint_mask is None:
                raise FileNotFoundError(f"Could not read keypoint mask: {keypoint_mask_path}")
            self.road_masks.append(road_mask)
            self.keypoint_masks.append(keypoint_mask)
            height, width = rgb.shape[:2]
            self.image_shapes.append((height, width))
            graph_label_generator = GraphLabelGenerator(config, gt_graph_adj, coord_transform)
            self.graph_label_generators.append(graph_label_generator)
            self.tile_ids.append(tile_idx)
            tile_meta = metadata_tiles.get(tile_idx, {}) if self.config.DATASET == 'cityscale' else {}
            self.tile_metadata.append(
                {
                    'domain': str(tile_meta.get('domain', 'default')),
                    'region': str(tile_meta.get('region', tile_idx)),
                    'time': str(tile_meta.get('time', tile_meta.get('input_group', 'default'))),
                }
            )
            self.train_patch_count += (
                get_eval_patches_per_axis(width, self.SAMPLE_MARGIN, self.config.PATCH_SIZE)
                * get_eval_patches_per_axis(height, self.SAMPLE_MARGIN, self.config.PATCH_SIZE)
            )

        if not self.rgbs:
            raise RuntimeError('No valid dataset samples were loaded. Check the generated custom dataset and split config.')

        self.tile_id_to_local_idx = {tile_id: local_idx for local_idx, tile_id in enumerate(self.tile_ids)}
        self._build_train_sampling_index()

        if not self.is_train:
            self.eval_patches = []
            for i, (height, width) in enumerate(self.image_shapes):
                eval_patches_per_width = get_eval_patches_per_axis(width, self.SAMPLE_MARGIN, self.config.PATCH_SIZE)
                eval_patches_per_height = get_eval_patches_per_axis(height, self.SAMPLE_MARGIN, self.config.PATCH_SIZE)
                self.eval_patches += get_patch_info_one_img(
                    i,
                    width,
                    height,
                    self.SAMPLE_MARGIN,
                    self.config.PATCH_SIZE,
                    eval_patches_per_width,
                    eval_patches_per_height,
                )

    def _build_train_sampling_index(self):
        self.domain_region_time_indices = {}
        for img_idx, item in enumerate(self.tile_metadata):
            domain = item['domain']
            region = item['region']
            time_name = item['time']
            self.domain_region_time_indices.setdefault(domain, {}).setdefault(region, {}).setdefault(time_name, []).append(img_idx)

    def _sample_train_image_index(self):
        domains = sorted(self.domain_region_time_indices)
        old_domains_value = self.config.get('OLD_DOMAIN_NAMES', ['old', 'default'])
        if isinstance(old_domains_value, str):
            old_domains = {item.strip() for item in old_domains_value.split(',') if item.strip()}
        else:
            old_domains = {str(item) for item in old_domains_value}
        old_available = [domain for domain in domains if domain in old_domains]
        new_available = [domain for domain in domains if domain not in old_domains]
        new_ratio = float(self.config.get('NEW_DOMAIN_RATIO', 0.6))
        if old_available and new_available:
            domain_pool = new_available if np.random.rand() < new_ratio else old_available
        else:
            domain_pool = domains
        domain = str(np.random.choice(domain_pool))
        regions = self.domain_region_time_indices[domain]
        region = str(np.random.choice(sorted(regions)))
        times = regions[region]
        time_name = str(np.random.choice(sorted(times)))
        return int(np.random.choice(times[time_name]))

    def sampling_summary(self):
        return {
            domain: {
                region: {time_name: len(indices) for time_name, indices in times.items()}
                for region, times in regions.items()
            }
            for domain, regions in self.domain_region_time_indices.items()
        }

    def __len__(self):
        if self.is_train:
            return max(1, self.train_patch_count)
        else:
            return len(self.eval_patches)

    def __getitem__(self, idx):
        
        if self.is_train:
            img_idx = self._sample_train_image_index()
            height, width = self.image_shapes[img_idx]
            sample_min_x, sample_max_x = get_axis_patch_range(width, self.SAMPLE_MARGIN, self.config.PATCH_SIZE)
            sample_min_y, sample_max_y = get_axis_patch_range(height, self.SAMPLE_MARGIN, self.config.PATCH_SIZE)
            begin_x = np.random.randint(low=sample_min_x, high=sample_max_x + 1)
            begin_y = np.random.randint(low=sample_min_y, high=sample_max_y + 1)
            end_x, end_y = begin_x + self.config.PATCH_SIZE, begin_y + self.config.PATCH_SIZE
        else:
            # Returns eval patch
            img_idx, (begin_x, begin_y), (end_x, end_y) = self.eval_patches[idx]  
            img_idx = int(img_idx)
        # Crop patch imgs and masks
        rgb_patch = self.rgbs[img_idx][begin_y:end_y, begin_x:end_x, :]
        keypoint_mask_patch = self.keypoint_masks[img_idx][begin_y:end_y, begin_x:end_x]
        road_mask_patch = self.road_masks[img_idx][begin_y:end_y, begin_x:end_x]    
        # Augmentation
        rot_index = 0
        if self.is_train:
            rot_index = np.random.randint(0, 4)
            # CCW
            rgb_patch = np.rot90(rgb_patch, rot_index, [0,1]).copy()
            keypoint_mask_patch = np.rot90(keypoint_mask_patch, rot_index, [0, 1]).copy()
            road_mask_patch = np.rot90(road_mask_patch, rot_index, [0, 1]).copy()       
            if self.config.get('AUG_RESOLUTION', False):
                rgb_patch = apply_resolution_aug(rgb_patch, self.config)
            if self.config.get('AUG_RADIOMETRIC', False):
                rgb_patch = apply_radiometric_aug(rgb_patch, self.config)
            if self.config.get('AUG_SENSOR_STYLE', False):
                rgb_patch = apply_sensor_style_aug(rgb_patch, self.config)
        # Sample graph labels from patch
        patch = ((begin_x, begin_y), (end_x, end_y))
        # points are img (x, y) inside the patch.
        graph_points, topo_samples = self.graph_label_generators[img_idx].sample_patch(patch, rot_index)       
        pairs, connected, valid = zip(*topo_samples)  
        # rgb: [H, W, 3] 0-255
        # masks: [H, W] 0-1
        return {
            'rgb': torch.tensor(rgb_patch, dtype=torch.float32),
            'keypoint_mask': torch.tensor(keypoint_mask_patch, dtype=torch.float32) / 255.0,
            'road_mask': torch.tensor(road_mask_patch, dtype=torch.float32) / 255.0,
            
            'graph_points': torch.tensor(graph_points, dtype=torch.float32),
            'pairs': torch.tensor(pairs, dtype=torch.int32),
            'connected': torch.tensor(connected, dtype=torch.bool),
            'valid': torch.tensor(valid, dtype=torch.bool),
        }

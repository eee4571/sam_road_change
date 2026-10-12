"""Compare extracted road length using the production SAMRoad/TopoNet path.

More extracted centerline length is the selection objective, not accuracy.
Only local RGB windows are inferred; no whole-scene products are generated.
"""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


def graph_metrics(nodes, edges, valid, pixel_size=.5):
    """Length inside valid central pixels, independent of width/segmentation.

    Nodes are row/column coordinates as returned by infer_one_img. Deduplicate
    undirected edges; subdivide at half-pixel spacing to exclude nodata and the
    64-pixel context margin without counting a whole crossing edge as valid.
    """
    inside = valid.copy()
    inside[:64] = False; inside[-64:] = False
    inside[:, :64] = False; inside[:, -64:] = False
    length_px = 0.
    unique = {tuple(sorted((int(a),int(b)))) for a,b in edges if a != b}
    for a,b in unique:
        start, end = np.asarray(nodes[a],float), np.asarray(nodes[b],float)
        length = float(np.linalg.norm(end-start))
        if not np.isfinite(length):
            raise ValueError('模型道路坐标无效')
        count = max(1,int(np.ceil(length*2)))
        positions = start + ((np.arange(count)+.5)/count)[:,None]*(end-start)
        rows, cols = np.floor(positions+.5).astype(int).T
        on_image = (rows>=0)&(rows<inside.shape[0])&(cols>=0)&(cols<inside.shape[1])
        accepted = inside[rows[on_image],cols[on_image]].sum()
        length_px += length*float(accepted)/count
    area = int(inside.sum())*pixel_size**2
    length_m = length_px*pixel_size
    return dict(score=length_m/max(area,1.)*1_000_000., length_m=length_m,
                valid_area_m2=area, edge_count=len(unique))


class RoadModelProbe:
    """Load once, lazily, and reuse for all sampled periods and areas."""
    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[3])
        self.config_path = self.root / 'runtime/config/samroad_inference.yaml'
        self.checkpoint = self.root / 'runtime/model/samroad/samroad.ckpt'
        self.sam_checkpoint = self.root / 'runtime/model/samroad/sam_vit_b_01ec64.pth'
        self.net = None

    def identity(self):
        paths = [self.config_path, self.checkpoint, self.sam_checkpoint, Path(__file__),
                 *[Path(__file__).with_name(name) for name in
                   ('modelinfer.py','inferencer.py','graph_extraction.py','fast_probability.py','topology_aggregation.py')]]
        return [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in paths]

    def _load(self):
        import torch
        # SAMRoad's existing modules use local absolute imports. This dedicated
        # backend process has no host GUI modules to collide with those names.
        sys.path.insert(0, str(Path(__file__).parent))
        from utils import load_config
        from modelinfer import SAMRoadplus
        config = load_config(self.config_path)
        if int(config.PATCH_SIZE) != 512:
            raise ValueError('当前自动推荐需要 512 像素的道路模型配置，请手动指定参考期')
        config.SAM_CKPT_PATH = str(self.sam_checkpoint)
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(min(4, torch.get_num_threads()))
        net = SAMRoadplus(config)
        checkpoint = torch.load(self.checkpoint, map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'], strict=True)
        self.net = net.eval().to(self.device)

    def evaluate(self, rgb, valid):
        if self.net is None:
            self._load()
        import inferencer
        inferencer.args = SimpleNamespace(device=str(self.device), execution_profile='fast')
        # Reuse actual production graph extraction and TopoNet, including its
        # fixed configured threshold policy. Config is copied because profile
        # resolution changes it per image inside the existing inferencer.
        result = inferencer.infer_one_img(self.net, rgb, copy.deepcopy(self.config))
        return graph_metrics(result[0], result[1], valid)

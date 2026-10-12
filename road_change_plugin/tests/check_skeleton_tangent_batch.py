"""Exact equivalence of batched neighbor queries, without loading a model."""
import ast
from pathlib import Path
import unittest
import numpy as np
from sklearn.neighbors import KDTree


source = Path(__file__).resolve().parents[1] / 'code/engine/samroad/graph_extraction.py'
tree = ast.parse(source.read_text(encoding='utf8'))
function = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == 'estimate_skeleton_tangents')
namespace = dict(np=np, KDTree=KDTree)
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
estimate = namespace['estimate_skeleton_tangents']


def per_point_baseline(points, radius):
    if not len(points):
        return np.empty((0,2),dtype=np.float32)
    tree = KDTree(points)
    output = np.zeros((len(points),2), dtype=np.float32)
    for i,point in enumerate(points):
        neighbors = points[tree.query_radius(point[None,:],r=radius)[0]].astype(np.float32)
        if len(neighbors)<2:
            output[i] = (1.,0.)
            continue
        centered = neighbors-np.mean(neighbors,axis=0,keepdims=True)
        _,vectors = np.linalg.eigh(centered.T@centered)
        tangent = vectors[:,-1]
        output[i] = tangent/max(float(np.linalg.norm(tangent)),1e-6)
    return output


class TangentBatchTests(unittest.TestCase):
    def test_exact_values_for_lines_junctions_and_dense_points(self):
        rng = np.random.default_rng(7)
        x = np.arange(1200)
        scenes = [np.empty((0,2)), np.array([[0,0]]),
                  np.concatenate([np.column_stack([x,np.zeros_like(x)]),
                                  np.column_stack([x,np.full_like(x,4)]),
                                  np.column_stack([np.full_like(x,200),x])]),
                  rng.uniform(0,100,(3100,2))]
        for points in scenes:
            points = points.astype(np.float32)
            np.testing.assert_array_equal(estimate(points,5.), per_point_baseline(points,5.))

    def test_neighbor_storage_and_dispatch_are_bounded(self):
        sizes = []
        class RecordingTree:
            def __init__(self,points): self.tree=KDTree(points)
            def query_radius(self,points,**kwargs):
                sizes.append(len(points))
                return self.tree.query_radius(points,**kwargs)
        original = namespace['KDTree']
        namespace['KDTree'] = RecordingTree
        try:
            estimate(np.column_stack([np.arange(2500),np.zeros(2500)]).astype(np.float32))
        finally:
            namespace['KDTree'] = original
        self.assertEqual(sizes,[1024,1024,452])


if __name__ == '__main__':
    unittest.main()

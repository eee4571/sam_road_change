"""Order-preserving vectorized aggregation of valid TopoNet edge votes."""
import numpy as np


def aggregate_votes(scores, pairs, valid, index_maps, edge_scores, edge_counts):
    bi, si, pi = np.nonzero(np.asarray(valid))
    if not len(bi):
        return
    local = np.asarray(pairs)[bi, si, pi]
    size = max(max(row,default=-1)+1 if isinstance(row,dict) else len(row) for row in index_maps)
    lookup = np.full((len(index_maps), size), -1, dtype=np.int64)
    for i, row in enumerate(index_maps):
        if isinstance(row,dict):
            lookup[i,list(row)] = list(row.values())
        else:
            lookup[i, :len(row)] = row
    edges = lookup[bi[:, None], local]
    if np.any(edges < 0):
        raise KeyError('Valid TopoNet pair has no full-graph index')
    values = np.asarray(scores)[bi, si, pi]
    assert np.all((values >= 0.) & (values <= 1.))
    unique, first, inverse = np.unique(edges, axis=0, return_index=True, return_inverse=True)
    # Match the original Python/NumPy scalar accumulator dtype and vote order.
    dtype = np.asarray(0.0+values[0]).dtype
    totals = np.array([edge_scores.get(tuple(edge), 0.) for edge in unique], dtype=dtype)
    counts = np.array([edge_counts.get(tuple(edge), 0.) for edge in unique], dtype=float)
    np.add.at(totals, inverse, values)
    np.add.at(counts, inverse, 1.)
    for index in np.argsort(first):
        key = tuple(unique[index])
        edge_scores[key] = totals[index]
        edge_counts[key] = counts[index]

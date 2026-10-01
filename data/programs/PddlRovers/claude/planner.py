"""Task planner: choose stand points and orders for the two rovers."""
import itertools
import numpy as np
import geom
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

HOME = [np.array([1.0, -1.75]), np.array([-1.0, -1.75])]


class GridGraph:
    """8-connected unit-cost grid graph over free cells (L-inf metric)."""

    def __init__(self, model, res=geom.GRID_RES, clearance=0.0):
        free = model.build_grid(res=res, clearance=clearance)
        n = model.n
        self.n = n
        self.res = res
        self.xs = model.xs
        self.free = free
        idx = -np.ones((n, n), dtype=np.int64)
        fi, fj = np.nonzero(free)
        idx[fi, fj] = np.arange(len(fi))
        self.idx = idx
        self.cells = np.stack([fi, fj], axis=1)
        self.pts = np.stack([self.xs[fi], self.xs[fj]], axis=1)
        rows, cols = [], []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                a = fi + di
                b = fj + dj
                m = (a >= 0) & (a < n) & (b >= 0) & (b < n)
                m2 = m.copy()
                m2[m] = free[a[m], b[m]]
                rows.append(idx[fi[m2], fj[m2]])
                cols.append(idx[a[m2], b[m2]])
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.ones(len(rows), dtype=np.float32)
        self.graph = csr_matrix((data, (rows, cols)), shape=(len(fi), len(fi)))

    def node_of(self, p):
        """Nearest free node to point p."""
        d = np.sum((self.pts - np.asarray(p, float)[None, :2]) ** 2, axis=1)
        return int(np.argmin(d))

    def dists(self, sources):
        d, pred = dijkstra(self.graph, indices=sources, unweighted=True,
                           return_predecessors=True, directed=False)
        return np.atleast_2d(d), np.atleast_2d(pred)

    def path(self, pred_row, target):
        """List of node ids from source to target using predecessor row."""
        out = [target]
        cur = target
        while pred_row[cur] >= 0:
            cur = int(pred_row[cur])
            out.append(cur)
        out.reverse()
        return out


def region_nodes(graph, mask):
    return np.nonzero(mask)[0]


def objective_view_mask(model, graph, xy, rng=None, pad=None):
    rng = geom.IMG_RANGE if rng is None else rng
    d = np.hypot(graph.pts[:, 0] - xy[0], graph.pts[:, 1] - xy[1])
    m = d <= rng - 1e-3
    if not np.any(m):
        return m
    vis = np.zeros(len(graph.pts), dtype=bool)
    idxs = np.nonzero(m)[0]
    boxes = model.sight_boxes
    if pad is not None and len(boxes):
        boxes = boxes.copy()
        boxes[:, 2:4] += pad - geom.SIGHT_PAD
    old = model.sight_boxes
    model.sight_boxes = boxes
    try:
        vis[idxs] = model.visible_many(graph.pts[idxs], xy)
    finally:
        model.sight_boxes = old
    return m & vis


def sample_mask(graph, xy, rng=None):
    rng = geom.SAMPLE_RANGE - 0.06 if rng is None else rng
    d = np.hypot(graph.pts[:, 0] - xy[0], graph.pts[:, 1] - xy[1])
    return d <= rng - 1e-3


def send_mask(model, graph):
    return model.can_send_many(graph.pts)


def pick_reps(graph, nodes, k, anchors_pts=()):
    """Pick up to k representative nodes: farthest-point sampling seeded by
    the nodes closest to each anchor point."""
    if len(nodes) == 0:
        return []
    pts = graph.pts[nodes]
    chosen = []
    for a in anchors_pts:
        i = int(np.argmin(np.sum((pts - np.asarray(a, float)[None, :2]) ** 2, axis=1)))
        if i not in chosen:
            chosen.append(i)
        if len(chosen) >= k:
            return [int(nodes[i]) for i in chosen]
    if not chosen:
        chosen.append(0)
    while len(chosen) < k and len(chosen) < len(nodes):
        d = np.min(np.linalg.norm(pts[:, None, :] - pts[chosen][None, :, :], axis=2), axis=1)
        i = int(np.argmax(d))
        if d[i] <= 1e-9:
            break
        chosen.append(i)
    return [int(nodes[i]) for i in chosen]

import numpy as np
from scipy import ndimage
def base_reach_labels(obs, r, res=0.01, W=2.5):
    n = int(round(W / res))
    xs = (np.arange(n) + 0.5) * res
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    reff = r - res * 0.75
    free = (X > reff) & (X < W - reff) & (Y > reff) & (Y < W - reff)
    for k in range(obs.n):
        c = obs.C[k]; R = obs.R[k]
        i0 = max(0, int((c[0] - R - r) / res) - 1); i1 = min(n, int((c[0] + R + r) / res) + 2)
        j0 = max(0, int((c[1] - R - r) / res) - 1); j1 = min(n, int((c[1] + R + r) / res) + 2)
        if i0 >= i1 or j0 >= j1: continue
        px = X[i0:i1, j0:j1] - obs.origin[k][0]; py = Y[i0:i1, j0:j1] - obs.origin[k][1]
        u = obs.A[k][0][0] * px + obs.A[k][0][1] * py
        v = obs.A[k][1][0] * px + obs.A[k][1][1] * py
        du = np.maximum(0, np.maximum(-u, u - obs.len[k][0]))
        dv = np.maximum(0, np.maximum(-v, v - obs.len[k][1]))
        free[i0:i1, j0:j1] &= (du * du + dv * dv) >= reff * reff
    lab, _ = ndimage.label(free, structure=np.ones((3, 3)))
    return lab, res
def cell(lab, res, p):
    n = lab.shape[0]
    i = min(n - 1, max(0, int(p[0] / res))); j = min(n - 1, max(0, int(p[1] / res)))
    return lab[i, j]

"""Search redundant collision-free shallow-east grasps (diagnostic only)."""
import math
import sys

import numpy as np
from scipy.optimize import least_squares

from env_client import make_env
from approach import GeneratedApproach
from probe_grasp import fk
from search_grasp_grid_local import setup, vals, g, move

LO = np.array([-.715, .08, -.8, -2.321, -math.pi, -2.094, -math.pi])
HI = np.array([2.285, 1.396, 3.9, 0., math.pi, 0., math.pi])


def candidates(seed, n=120):
    e, s, p = setup(seed)
    base = vals(s, p)[:3]
    green = np.array([g(s, "green0", "pose_" + x) for x in "xyz"])
    # FK calibration from successful blocker grasps: physical block lies at
    # model-tool + this nearly seed-invariant displacement.
    goal = green - np.array([-.0601, -.0372, -.07527])
    e.close()
    rng = np.random.default_rng(9000 + seed)
    sols = []
    grids = [(q5, q7) for q5 in np.linspace(-math.pi, math.pi, 9)
             for q7 in np.linspace(-math.pi, math.pi, 9)]
    for k, (q5, q7) in enumerate(grids):
        # Position is essential; horizontal distal-link orientation and a
        # bent elbow are soft preferences only.
        free = [0, 1, 2, 3, 5]
        def unpack(x):
            q = np.empty(7); q[free] = x; q[4] = q5; q[6] = q7
            return q
        def fun(x):
            q = unpack(x)
            T = fk(base, q)
            return np.r_[30. * (T[:3, 3] - goal),
                         .004 * (q - p.Q)]
        z = least_squares(fun, rng.uniform(LO[free], HI[free]),
                          bounds=(LO[free], HI[free]),
                          max_nfev=500)
        q = unpack(z.x)
        if np.linalg.norm(fk(base, q)[:3, 3] - goal) < .012 and \
                min([np.linalg.norm(q-x) for x in sols] or [99.]) > .15:
            sols.append(q)
    return base, goal, sorted(sols, key=lambda q: np.linalg.norm(q-p.Q))


def test(seed):
    base, goal, sols = candidates(seed)
    print("generated", len(sols), "goal", goal, flush=True)
    for k, q in enumerate(sols):
        e, s, p = setup(seed)
        start = vals(s, p)
        # First change redundant wrist/elbow coordinates at lifted shoulder,
        # then approach the endpoint slowly while closing.
        pre = start.copy()
        pre[3:] = q
        pre[4] = start[4]
        s, _ = move(e, s, p, pre, False, 70, .035)
        s, ok = move(e, s, p, np.r_[base, q], True, 70, .018)
        actual = vals(s, p)
        if ok:
            print("HIT", seed, k, "q", q.tolist(), "actual", actual.tolist(),
                  "fk", fk(actual[:3], actual[3:])[:3, 3].tolist(), flush=True)
            e.close(); return
        if k % 10 == 0:
            print("try", k, "q", np.round(q, 3).tolist(),
                  "actual", np.round(actual[3:], 3).tolist(),
                  "err", np.round(fk(actual[:3], actual[3:])[:3, 3]-goal, 3).tolist(),
                  flush=True)
        e.close()
    print("NONE", seed, len(sols), flush=True)


if __name__ == "__main__":
    test(int(sys.argv[1]) if len(sys.argv) > 1 else 101)

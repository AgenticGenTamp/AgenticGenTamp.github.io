"""Local tool-frame/Jacobian probe for shallow-east failures (research only)."""
import math
import sys

import numpy as np

from env_client import make_env
from approach import GeneratedApproach


def g(s, n, f):
    return float(s.get(s.get_object_from_name(n), f))


def quat_matrix(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def measure(s):
    p = np.array([g(s, "robot", "grasp_tf_"+x) for x in "xyz"])
    q = np.array([g(s, "robot", "grasp_tf_q"+x) for x in "xyzw"])
    b = np.array([g(s, "green0", "pose_"+x) for x in "xyz"])
    R = quat_matrix(q)
    return p, R, R.T @ (b-p)


def setup(seed):
    e = make_env(); s, info = e.reset(seed=seed)
    p = GeneratedApproach(e.action_space, e.observation_space, {}); p.reset(s, info)
    for k in range(200):
        if p.stage == 5:
            break
        s, *_ = e.step(p.get_action(s))
    # Execute stage 5 until a rejected command leaves us stationary.
    for k in range(40):
        old = np.array([g(s, "robot", "base_x"), g(s, "robot", "base_y"),
                        *[g(s, "robot", "joint_"+str(j)) for j in range(1, 8)]])
        s, *_ = e.step(p.get_action(s))
        new = np.array([g(s, "robot", "base_x"), g(s, "robot", "base_y"),
                        *[g(s, "robot", "joint_"+str(j)) for j in range(1, 8)]])
        if np.max(abs(new-old)) < 1e-7:
            break
    return e, s, p


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 101
    e, s, p = setup(seed)
    xyz, R, local = measure(s)
    print("seed", seed, "stage", p.stage, "base", p.robot(s), "theta", g(s, "robot", "base_rot"))
    print("q", [round(g(s, "robot", "joint_"+str(j)), 6) for j in range(1, 8)])
    print("tool", xyz, "local block", local, "axes\n", R)
    e.close()

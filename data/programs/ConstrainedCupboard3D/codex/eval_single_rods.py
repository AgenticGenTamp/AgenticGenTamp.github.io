"""Concise full-horizon audit of the current policy on one-rod episodes."""
import concurrent.futures
import math
import sys

import numpy as np

from approach import GeneratedApproach
from env_client import make_env


def xyz(state, obj):
    return np.array([float(state.get(obj, f)) for f in ("x", "y", "z")])


def run(seed):
    env = make_env()
    state, info = env.reset(seed=seed, options={"object_count": 1})
    policy = GeneratedApproach(env.action_space, env.observation_space,
                               env.make_primitives())
    policy.reset(state, info)
    rod = policy.rods[0]
    p0 = xyz(state, rod)
    target_y = policy.slot_y[0]
    total = 0.0
    positive = 0.0
    best = -1e9
    term = trunc = False
    for step in range(env.max_steps):
        state, reward, term, trunc, info = env.step(policy.get_action(state))
        total += reward
        positive += max(0.0, reward)
        best = max(best, reward)
        if term or trunc:
            break
    p1 = xyz(state, rod)
    yaw = 2 * math.atan2(float(state.get(rod, "qz")),
                         float(state.get(rod, "qw")))
    # For one rod the three cupboard dividers at y=-.1,0,.1 expose
    # two plausible gap centers.  Report distance to the nearer center.
    fix_type = env.observation_space.get_type("mujoco_fixture")
    fy = sorted(float(state.get(o, "y")) for o in state.get_objects(fix_type))
    gaps = [(a + b) / 2 for a, b in zip(fy[:-1], fy[1:])]
    slot_dist = min(math.hypot(p1[0] - 2.0, p1[1] - y) for y in gaps)
    env.close()
    return (seed, step + 1, p0, p1, yaw, target_y, slot_dist,
            total, positive, best, term, trunc)


if __name__ == "__main__":
    seeds = [int(v) for v in sys.argv[1:]] or list(range(4))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(seeds)) as pool:
        results = list(pool.map(run, seeds))
    for seed, steps, p0, p1, yaw, ty, dist, total, pos, best, term, trunc in results:
        print(seed, steps, "p0", np.round(p0, 3), "p1", np.round(p1, 3),
              "d", np.round(p1-p0, 3), "slot_y", round(ty, 3),
              "yerr0/1", round(abs(p0[1]-ty), 3), round(abs(p1[1]-ty), 3),
              "yaw", round(yaw, 3), "near_slot_dist", round(dist, 3),
              "ret/pos/best", round(total, 3), round(pos, 3), round(best, 3),
              "done", term, trunc, flush=True)

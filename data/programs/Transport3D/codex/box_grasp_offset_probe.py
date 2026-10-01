"""Vary the final known box grasp and report object-in-hand transforms."""
import math
import numpy as np
from env_client import make_env
from probe_grasp_structured import command, val

lo = np.array([0., -.35, -math.pi, -2.5, 0., -.87, math.pi/2])
hi = lo + np.array([5.2, 2.41, 2.058, 2.66, 5.2, 2.23, 6.77])
rng = np.random.default_rng(0); route = []
for _ in range(28):
    q = rng.uniform(lo, hi); radius = rng.uniform(.05, .95); angle = rng.uniform(-math.pi, math.pi)
    route.append((q, radius, angle, rng.uniform(-math.pi, math.pi)))

env = make_env()
variants = [(j, d) for j in range(7) for d in (-.08, -.04, .04, .08)]
variants.insert(0, (-1, 0.))
for j, delta in variants:
    s, _ = env.reset(seed=0, options={'object_count': 0})
    tx, ty = val(s, 'box0', 'pose_x'), val(s, 'box0', 'pose_y')
    # Preserve collision-selecting waypoints, but only close at the final pose.
    for idx in (24, 26):
        q, radius, angle, rot = route[idx]
        base = [tx+radius*math.cos(angle), ty+radius*math.sin(angle), rot]
        s = command(env, s, base, q, 1., 35)
    q, radius, angle, rot = route[27]
    q = q.copy()
    if j >= 0: q[j] += delta
    base = [tx+radius*math.cos(angle), ty+radius*math.sin(angle), rot]
    s = command(env, s, base, q, 1., 35)
    s = command(env, s, base, q, -1., 2)
    held = val(s, 'robot', 'grasp_active') > .5
    tf = [val(s, 'robot', 'grasp_tf_'+c) for c in 'xyz']
    quat = [val(s, 'robot', 'grasp_tf_q'+c) for c in 'xyzw']
    print(j, delta, 'held', held, 'tf', np.round(tf, 4), 'quat', np.round(quat, 3), flush=True)
env.close()

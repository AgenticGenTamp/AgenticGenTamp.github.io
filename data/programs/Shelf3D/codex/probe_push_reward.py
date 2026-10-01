"""Use the mobile base as a coarse probe of reward versus cube location."""

import argparse
import numpy as np

from env_client import make_env


p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=123)
p.add_argument("--count", type=int, default=1)
p.add_argument("--axis", type=int, default=0)
p.add_argument("--sign", type=float, default=1.0)
p.add_argument("--steps", type=int, default=40)
p.add_argument("--offset-y-steps", type=int, default=0)
args = p.parse_args()

env = make_env()
state, info = env.reset(seed=args.seed, options={"object_count": args.count})
action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
action[args.axis] = args.sign * 0.1
if args.offset_y_steps:
    offset = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    offset[1] = np.sign(args.offset_y_steps) * .1
    for _ in range(abs(args.offset_y_steps)):
        state, _, _, _, _ = env.step(offset)
for step in range(args.steps + 1):
    cube = state.get_object_from_name("cube1")
    robot = state.get_object_from_name("robot")
    xyz = tuple(round(float(state.get(cube, f)), 3) for f in ("x", "y", "z"))
    base = tuple(round(float(state.get(robot, f)), 3) for f in
                 ("pos_base_x", "pos_base_y", "pos_base_rot"))
    if step == 0:
        print(step, xyz, base, None, False, False)
    else:
        print(step, xyz, base, reward, terminated, truncated)
    if step == args.steps or (step and (terminated or truncated)):
        break
    state, reward, terminated, truncated, info = env.step(action)
env.close()

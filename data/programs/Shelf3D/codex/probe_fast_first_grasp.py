"""Measure shortened first-grasp schedules on multi-object layouts.

Usage: python probe_fast_first_grasp.py retreat lower align close lift [seeds...]
The probe deliberately duplicates only the policy's verified grasp prefix so
experiments cannot mutate the submitted approach.
"""

import math
import sys

import numpy as np

from env_client import make_env


HOME = np.array([0.0, -0.3491, math.pi, -2.5482, 0.0, -0.8727,
                 math.pi / 2])
GROUND = np.array([0.0, 2.24, 2.945, -1.0, -0.982, 0.20, 1.30])
LIFT = np.array([0.0, 1.45, 2.945, -1.0, -0.982, 0.20, 1.30])


def run(seed, durations):
    env = make_env()
    state, _ = env.reset(seed=seed, options={"object_count": 2})
    names = sorted(n for n in state.get_object_names() if n.startswith("cube"))
    cube = state.get_object_from_name(names[0])
    pick = np.array([state.get(cube, "x"), state.get(cube, "y")])
    contact = np.array([pick[0] - .609, pick[1] - .054, 0.0])
    retreat = contact.copy()
    retreat[0] -= .195
    poses = (HOME, GROUND, GROUND, GROUND, LIFT)
    bases = (retreat, retreat, contact, contact, contact)
    grips = (0.0, 0.0, 0.0, .6, .6)
    try:
        for stage, duration in enumerate(durations):
            for _ in range(duration):
                robot = state.get_object_from_name("robot")
                base = np.array([state.get(robot, "pos_base_x"),
                                 state.get(robot, "pos_base_y"),
                                 state.get(robot, "pos_base_rot")])
                joints = np.array([state.get(robot, "pos_arm_joint%d" % i)
                                   for i in range(1, 8)])
                action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
                action[:2] = np.clip((bases[stage][:2] - base[:2]) / .87, -.1, .1)
                angle = (bases[stage][2] - base[2] + math.pi) % (2*math.pi) - math.pi
                action[2] = np.clip(angle / .87, -.1, .1)
                action[3:10] = np.clip(.35 * (poses[stage] - joints), -.1, .1)
                action[10] = grips[stage]
                state, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break
        xyz = tuple(float(state.get(cube, f)) for f in ("x", "y", "z"))
        return xyz, xyz[2] > .08
    finally:
        env.close()


if __name__ == "__main__":
    durations = tuple(map(int, sys.argv[1:6]))
    seeds = tuple(map(int, sys.argv[6:])) or tuple(range(5))
    results = [(s,) + run(s, durations) for s in seeds]
    print(durations, results)

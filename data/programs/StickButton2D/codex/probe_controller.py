"""Evaluate simple black-box geometric policies without touching approach.py."""

import argparse
import math

import numpy as np

from env_client import make_env


def value(state, obj, feature):
    return float(state.get(obj, feature))


def red_buttons(state, circle_type):
    return [
        obj for obj in state.get_objects(circle_type)
        if value(state, obj, "color_r") > value(state, obj, "color_g")
    ]


def run(seed, limit=1000, verbose=False):
    env = make_env()
    state, info = env.reset(seed=seed)
    circle_type = env.observation_space.get_type("circle")
    robot_type = env.observation_space.get_type("crv_robot")
    total = len(state.get_objects(circle_type))
    blocked = 0
    last_xy = None
    for step in range(limit):
        remaining = red_buttons(state, circle_type)
        if not remaining:
            # Usually termination occurred in the preceding transition.
            env.close()
            return step, True, total, blocked
        robot = state.get_objects(robot_type)[0]
        rx, ry = value(state, robot, "x"), value(state, robot, "y")
        target = min(
            remaining,
            key=lambda b: max(abs(value(state, b, "x") - rx),
                              abs(value(state, b, "y") - ry)),
        )
        dx = np.clip(value(state, target, "x") - rx, -0.05, 0.05)
        dy = np.clip(value(state, target, "y") - ry, -0.05, 0.05)
        action = np.asarray([dx, dy, 0.0, 0.0, 0.0], dtype=np.float32)
        state2, reward, terminated, truncated, info = env.step(action)
        robot2 = state2.get_objects(robot_type)[0]
        x2, y2 = value(state2, robot2, "x"), value(state2, robot2, "y")
        if abs(x2 - (rx + dx)) > 1e-4 or abs(y2 - (ry + dy)) > 1e-4:
            blocked += 1
            if verbose:
                print(seed, "blocked", step, (rx, ry), (dx, dy), (x2, y2))
        state = state2
        if terminated:
            env.close()
            return step + 1, True, total, blocked
        if truncated:
            break
    env.close()
    return limit, False, total, blocked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    failures = []
    records = []
    for seed in range(args.start, args.start + args.count):
        result = run(seed, verbose=args.verbose)
        records.append((seed,) + result)
        if not result[1]:
            failures.append(seed)
    print("failures", failures)
    print("max_steps", max(x[1] for x in records), "mean_steps", np.mean([x[1] for x in records]))
    print("counts", sorted(set(x[3] for x in records)), "blocked_total", sum(x[4] for x in records))
    print("slowest", sorted(records, key=lambda x: x[1], reverse=True)[:10])


if __name__ == "__main__":
    main()

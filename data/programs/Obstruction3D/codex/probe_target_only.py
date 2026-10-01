"""Probe exact target-block pick/place behavior on zero-obstruction seed 11.

This script is deliberately independent of approach.py.  It can either trace the
current controller or try a specified grasp offset/height and execute a circular
joint-1 transfer to the target region.
"""
import argparse
import math

import numpy as np

from env_client import make_env


SEED = 11
RADIUS = 0.656


def get(state, name, feature):
    return float(state.get(state.get_object_from_name(name), feature))


def action_to(state, base, q1, q4, q6=-0.87, grip=-1.0):
    action = np.zeros(11, dtype=np.float32)
    goals = ((0, "pos_base_x", base[0]), (1, "pos_base_y", base[1]),
             (3, "joint_1", q1), (4, "joint_2", 0.65),
             (6, "joint_4", q4), (8, "joint_6", q6))
    for index, feature, goal in goals:
        action[index] = np.clip(goal - get(state, "robot", feature), -0.2, 0.2)
    action[10] = grip
    return action


def circle_plan(source, destination):
    dx, dy = destination[0] - source[0], destination[1] - source[1]
    distance = math.hypot(dx, dy)
    midpoint = ((source[0] + destination[0]) / 2,
                (source[1] + destination[1]) / 2)
    height = math.sqrt(RADIUS * RADIUS - distance * distance / 4)
    centers = ((midpoint[0] - dy / distance * height,
                midpoint[1] + dx / distance * height),
               (midpoint[0] + dy / distance * height,
                midpoint[1] - dx / distance * height))
    base = min(centers, key=lambda point: point[0])
    start = -math.atan2(source[1] - base[1], source[0] - base[0])
    goal = start - (math.atan2(destination[1] - base[1], destination[0] - base[0])
                    - math.atan2(source[1] - base[1], source[0] - base[0]))
    return base, start, goal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", type=float, default=0.0)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--q4", type=float, default=-1.50)
    args = parser.parse_args()
    env = make_env()
    state, info = env.reset(seed=SEED)
    names = state.get_object_names()
    assert not [name for name in names if name.startswith("obstruction")], info
    source = tuple(get(state, "target_block", feature) for feature in ("pose_x", "pose_y"))
    destination = tuple(get(state, "target_region", feature) for feature in ("pose_x", "pose_y"))
    nominal_base, start, goal = circle_plan(source, destination)
    base = (nominal_base[0] + args.dx, nominal_base[1] + args.dy)
    print("geometry", source, destination, "nominal", nominal_base, start, goal,
          "trial", base, args.q4)

    # Settle at the candidate grasp pose, repeatedly commanding close.
    attached = False
    for step in range(18):
        state, reward, terminated, truncated, _ = env.step(
            action_to(state, base, start, args.q4))
        attached = get(state, "target_block", "grasp_active") > 0.5
        if attached:
            print("attached", step + 1, "block",
                  tuple(round(get(state, "target_block", f), 6)
                        for f in ("pose_x", "pose_y", "pose_z")))
            break
    if not attached:
        print("miss", "base", tuple(round(get(state, "robot", f), 6)
                                    for f in ("pos_base_x", "pos_base_y")),
              "joints", tuple(round(get(state, "robot", f"joint_{i}"), 6)
                              for i in range(1, 8)))
        env.close()
        return

    # Lift 10 cm by joint 6, swing around the fixed base, then lower and open.
    for phase, q1, q6, grip, count in (
            ("lift", start, -0.77, 0.0, 3),
            ("swing", goal, -0.77, 0.0, 8),
            ("lower", goal, -0.87, 0.0, 3),
            ("open", goal, -0.87, 1.0, 3)):
        for _ in range(count):
            state, reward, terminated, truncated, _ = env.step(
                action_to(state, base, q1, args.q4, q6, grip))
            if terminated or truncated:
                break
        print(phase, "done", terminated, truncated, "held",
              get(state, "target_block", "grasp_active"), "block",
              tuple(round(get(state, "target_block", f), 6)
                    for f in ("pose_x", "pose_y", "pose_z")))
        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":
    main()

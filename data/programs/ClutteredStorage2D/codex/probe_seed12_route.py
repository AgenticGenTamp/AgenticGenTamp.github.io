"""Probe recovery routes for seed 12's occupied-shelf approach."""

import sys
import numpy as np

from env_client import make_env
from approach import GeneratedApproach


def values(state, obj, fields):
    return tuple(round(float(state.get(obj, f)), 3) for f in fields)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "retreat_right"
    env = make_env()
    state, info = env.reset(seed=12, options={"object_count": 3})
    policy = GeneratedApproach(env.action_space, env.observation_space,
                               env.make_primitives())
    policy.reset(state, info)
    if mode in ("priority1", "stage_blocker"):
        policy.priority_name = "block1"
    robot = state.get_object_from_name("robot")
    shelf = state.get_object_from_name("shelf")
    print("shelf", values(state, shelf, ("x1", "y1", "width1", "height1")))
    print("robot", values(state, robot, ("x", "y", "theta", "arm_joint")))
    print("blocks", [(b.name, values(state, b, ("x", "y", "theta")))
                     for b in state.get_objects(policy.block_type)])
    recovery = 0
    staged = False
    for step in range(1000):
        action = policy.get_action(state)
        if mode == "stage_blocker" and staged and policy.stage == "clear_low":
            rx = float(state.get(robot, "x"))
            if rx < 2.60:
                action = np.array([.05, 0, 0, 0, 0], np.float32)
        if mode == "stage_blocker" and policy.target_name == "block1" and not staged \
                and policy.stage in ("lower_transport", "across_transport", "preplace"):
            blocker = state.get_object_from_name("block1")
            bx, by = (float(state.get(blocker, f)) for f in ("x", "y"))
            if abs(bx-2.0) < .015 and abs(by-1.15) < .015:
                action = np.array([0, 0, 0, 0, 0], np.float32)
                staged = True
                policy.stage = "choose"
                policy.target_name = None
            else:
                action = np.array([np.clip(2.0-bx, -.05, .05),
                                   np.clip(1.15-by, -.05, .05), 0, 0, 1], np.float32)
        if mode == "left_safe" and policy.clear_mode and policy.stage == "clear_across":
            policy.clear_side_x = 0.25
            policy.stage = "clear_escape_down"
            action = np.array([0, -.05, 0, 0, 0], np.float32)
        if mode == "left_side" and policy.clear_mode and policy.stage in (
                "clear_escape_down", "clear_escape_across", "clear_up"):
            policy.clear_side_x = 0.25
        if policy.stage == "clear_escape_down":
            recovery += 1
            if mode == "retreat_right" and recovery <= 10:
                action = np.array([.05, 0, 0, 0, 0], np.float32)
            elif mode == "retreat_diagonal" and recovery <= 10:
                action = np.array([.05, -.05, 0, 0, 0], np.float32)
        state, reward, terminated, truncated, step_info = env.step(action)
        if step % 50 == 49:
            target = (state.get_object_from_name(policy.target_name)
                      if policy.target_name is not None else robot)
            print(step + 1, policy.stage,
                  "r", values(state, robot, ("x", "y", "theta", "arm_joint", "vacuum")),
                  "b", values(state, target, ("x", "y", "theta")), "recovery", recovery)
        if terminated or truncated:
            print("DONE", step + 1, terminated, truncated)
            break
    else:
        print("FAIL", policy.stage)
    env.close()


if __name__ == "__main__":
    main()

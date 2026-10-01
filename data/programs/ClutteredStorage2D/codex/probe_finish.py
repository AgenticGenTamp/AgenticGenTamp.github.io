"""Probe shelf completion after the current policy's first placement."""

import math
import sys

import numpy as np

from approach import GeneratedApproach
from env_client import make_env


def f(state, obj, key):
    return float(state.get(obj, key))


def run(seed, mode):
    env = make_env()
    state, info = env.reset(seed=seed, options={"object_count": 1})
    policy = GeneratedApproach(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    intervention = False
    for step in range(300):
        robot = state.get_object_from_name("robot")
        block = state.get_object_from_name("block0")
        if not intervention:
            action = policy.get_action(state)
            if step > 50 and policy.stage == "choose":
                intervention = True
                print("INTERVENE", step)
        else:
            # The released block is just above an upward-pointing gripper.
            # Re-energize, then test small translations/extensions while held.
            t = step - intervention_step
            if t < 3:
                action = [0, 0, 0, 0, 1]
            elif mode == "up-base":
                action = [0, .02, 0, 0, 1] if t < 15 else [0, 0, 0, 0, 0]
            elif mode == "extend":
                action = [0, 0, 0, .02, 1] if t < 15 else [0, 0, 0, 0, 0]
            elif mode.startswith("arm"):
                goal = float(mode[3:])
                arm = f(state, robot, "arm_joint")
                if arm < goal - .005:
                    action = [0, 0, 0, min(.02, goal - arm), 1]
                else:
                    action = [0, 0, 0, 0, 0]
            elif mode.startswith("push"):
                goal = float(mode[4:])
                arm = f(state, robot, "arm_joint")
                by = f(state, block, "y")
                if arm < .395:
                    action = [0, 0, 0, .02, 1]
                elif by < goal:
                    action = [0, min(.02, goal - by), 0, 0, 1]
                else:
                    action = [0, 0, 0, 0, 0]
            elif mode == "down":
                action = [0, -.02, 0, 0, 1] if t < 15 else [0, 0, 0, 0, 0]
            else:
                raise ValueError(mode)
            action = np.asarray(action, dtype=np.float32)
        if intervention and "intervention_step" not in locals():
            intervention_step = step
        state, rew, term, trunc, out = env.step(action)
        if intervention:
            print(step + 1, mode,
                  "r", tuple(round(f(state, robot, k), 4) for k in ("x", "y", "theta", "arm_joint", "vacuum")),
                  "b", tuple(round(f(state, block, k), 4) for k in ("x", "y", "theta")),
                  "end", term, trunc)
        if term or trunc:
            break
    env.close()


def vertical(seed):
    """Rotate the carried block narrow-side-first, regrasp, and insert."""
    env = make_env()
    state, info = env.reset(seed=seed, options={"object_count": 1})
    policy = GeneratedApproach(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    phase = "policy"
    ticks = 0
    last_arm = None
    for step in range(400):
        robot = state.get_object_from_name("robot")
        block = state.get_object_from_name("block0")
        rx, ry, rt, arm = (f(state, robot, k) for k in ("x", "y", "theta", "arm_joint"))
        bx, by, bt = (f(state, block, k) for k in ("x", "y", "theta"))
        ticks += 1
        if phase == "policy":
            action = policy.get_action(state)
            if policy.stage == "drop_for_regrasp":
                # Override the policy's release: attachment is still live in
                # this observation, allowing a clean 90-degree rotation.
                phase, ticks = "rotate_vertical", 0
                action = np.asarray([0, 0, 0, 0, 1], np.float32)
        elif phase == "rotate_vertical":
            desired = (rt + (math.pi / 2 - bt) + math.pi) % (2 * math.pi) - math.pi
            err = (desired - rt + math.pi) % (2 * math.pi) - math.pi
            if abs((bt - math.pi / 2 + math.pi) % (2 * math.pi) - math.pi) < .015:
                phase, ticks = "separate_vertical", 0
                action = np.asarray([0, 0, 0, .1, 1], np.float32)
            else:
                action = np.asarray([0, 0, np.clip(err, -.196, .196), 0, 1], np.float32)
        elif phase == "separate_vertical":
            if arm < .58:
                action = np.asarray([0, 0, 0, .1, 1], np.float32)
            else:
                phase, ticks = "release_vertical", 0
                action = np.asarray([0, 0, 0, 0, 0], np.float32)
        elif phase == "release_vertical":
            if ticks < 3:
                action = np.asarray([0, 0, 0, 0, 0], np.float32)
            else:
                phase, ticks = "below", 0
                action = np.zeros(5, np.float32)
        elif phase == "below":
            ex, ey = bx - rx, by - .54 - ry
            et = (math.pi / 2 - rt + math.pi) % (2 * math.pi) - math.pi
            if max(abs(ex), abs(ey), abs(et), abs(arm - .2)) < .008:
                phase, ticks, last_arm = "extend_vertical", 0, arm
                action = np.zeros(5, np.float32)
            else:
                action = np.asarray([np.clip(ex, -.05, .05), np.clip(ey, -.05, .05),
                                     np.clip(et, -.196, .196), np.clip(.2-arm, -.1, .1), 0], np.float32)
        elif phase == "extend_vertical":
            stalled = last_arm is not None and abs(arm-last_arm) < .002
            last_arm = arm
            if stalled and ticks > 4:
                phase, ticks = "grip_vertical", 0
                action = np.asarray([0, 0, 0, 0, 1], np.float32)
            else:
                action = np.asarray([0, 0, 0, .1, 0], np.float32)
        elif phase == "grip_vertical":
            if ticks < 3:
                action = np.asarray([.01 if ticks == 2 else 0, 0, 0, 0, 1], np.float32)
            else:
                phase, ticks = "retract_vertical", 0
                action = np.asarray([0, 0, 0, -.1, 1], np.float32)
        elif phase == "retract_vertical":
            if arm > .21:
                action = np.asarray([0, 0, 0, -.1, 1], np.float32)
            else:
                phase, ticks = "to_slot", 0
                action = np.asarray([0, 0, 0, 0, 1], np.float32)
        elif phase == "to_slot":
            shelf = state.get_objects(env.observation_space.get_type("shelf"))[0]
            slotx = f(state, shelf, "x1") + f(state, shelf, "width1") / 2
            # Carry block center just below shelf's lower edge.
            ex, ey = slotx - bx, 2.57 - by
            if max(abs(ex), abs(ey)) < .008:
                phase, ticks = "insert_vertical", 0
                action = np.asarray([0, 0, 0, 0, 1], np.float32)
            else:
                action = np.asarray([np.clip(ex, -.05, .05), np.clip(ey, -.05, .05), 0, 0, 1], np.float32)
        else:  # insert_vertical
            # Move the whole assembly up; the narrow horizontal extent clears
            # the jambs. Release once the center is above the shelf lower edge.
            if by < 2.80:
                action = np.asarray([0, .02, 0, 0, 1], np.float32)
            else:
                action = np.asarray([0, 0, 0, 0, 0], np.float32)
        state, rew, term, trunc, out = env.step(action)
        if step % 5 == 0 or term:
            print(step + 1, phase, "r", tuple(round(f(state, robot, k), 3) for k in
                  ("x", "y", "theta", "arm_joint", "vacuum")), "b",
                  tuple(round(f(state, block, k), 3) for k in ("x", "y", "theta")), term)
        if term or trunc:
            print("END", step + 1, term, trunc)
            break
    env.close()


if __name__ == "__main__":
    if sys.argv[2] == "vertical":
        vertical(int(sys.argv[1]))
    else:
        run(int(sys.argv[1]), sys.argv[2])

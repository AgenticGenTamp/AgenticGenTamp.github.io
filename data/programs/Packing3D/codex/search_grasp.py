"""Prioritized contact/grasp search for one-part Packing3D seed 0."""

from __future__ import annotations

import math
import random
import sys

import numpy as np

from env_client import make_env


FEATURES = ["pos_base_x", "pos_base_y", "pos_base_rot"] + [
    f"joint_{i}" for i in range(1, 8)
] + ["finger_state", "grasp_active"]

KNOWN_JOINTS = np.array(
    [0.0, 0.26546195, -math.pi, -2.03142715, 0.0, -0.75871509, math.pi / 2]
)
KNOWN_TOOL_OFFSET = np.array([0.48035230, 0.00857803])


def robot_row(env, state):
    typ = env.observation_space.get_type("Kinematic3DRobot")
    obj = state.get_objects(typ)[0]
    return obj, np.array([state.get(obj, f) for f in FEATURES])


def fixture_xyz(env, state, name):
    obj = state.get_object_from_name(name)
    return np.array([state.get(obj, f) for f in ("pose_x", "pose_y", "pose_z")])


def step_and_check(env, state, action, tag, step_no):
    state, reward, terminated, truncated, info = env.step(
        np.asarray(action, dtype=np.float32)
    )
    robot, row = robot_row(env, state)
    part = state.get_object_from_name("part0")
    held = state.get(robot, "grasp_active") > 0.5 or state.get(part, "grasp_active") > 0.5
    if held:
        print("SUCCESS", tag, "step", step_no)
        print(" robot", dict(zip(FEATURES, row.tolist())))
        print(" part", fixture_xyz(env, state, "part0").tolist())
        print(" action", np.asarray(action).tolist())
        print(" reward/done/info", reward, terminated, truncated, info)
        return state, True
    return state, False


def run_trial(trial, rng):
    env = make_env()
    try:
        state, info = env.reset(seed=0, options={"object_count": 1})
        _, initial = robot_row(env, state)
        rack = fixture_xyz(env, state, "rack")
        part = fixture_xyz(env, state, "part0")
        if trial == 0:
            print("initial robot", dict(zip(FEATURES, initial.tolist())))
            print("rack", rack.tolist(), "part", part.tolist(), "offset", (part-rack).tolist())

            target_base = part[:2] - KNOWN_TOOL_OFFSET
            for step in range(6):
                action = np.zeros(11)
                _, current = robot_row(env, state)
                action[0:2] = np.clip(target_base - current[:2], -0.2, 0.2)
                action[3:10] = np.clip(KNOWN_JOINTS - current[3:10], -0.2, 0.2)
                action[10] = 1.0
                state, success = step_and_check(env, state, action, "known-approach", step)
                if success:
                    return True
            _, arrived = robot_row(env, state)
            print("known arrived", dict(zip(FEATURES, arrived.tolist())))
            action = np.zeros(11)
            action[10] = -1.0
            state, success = step_and_check(env, state, action, "known-close", 0)
            if success:
                return True

            # Triangle contact is narrower than cuboid contact. Search densely around
            # the reached cuboid pose without paying for a fresh environment per pose.
            local_rng = random.Random(7331)
            for candidate in range(500):
                target_joints = KNOWN_JOINTS.copy()
                target_joints += np.array(
                    [local_rng.uniform(-0.5, 0.5) for _ in range(7)]
                )
                target_xy = part[:2] - KNOWN_TOOL_OFFSET + np.array(
                    [local_rng.uniform(-0.12, 0.06), local_rng.uniform(-0.12, 0.12)]
                )
                for _ in range(3):
                    action = np.zeros(11)
                    _, current = robot_row(env, state)
                    action[0:2] = np.clip(target_xy - current[:2], -0.2, 0.2)
                    action[3:10] = np.clip(target_joints - current[3:10], -0.2, 0.2)
                    action[10] = 1.0
                    state, success = step_and_check(
                        env, state, action, "triangle-approach", candidate
                    )
                    if success:
                        return True
                action = np.zeros(11)
                action[10] = -1.0
                state, success = step_and_check(
                    env, state, action, "triangle-close", candidate
                )
                if success:
                    return True

        # The default gripper visually lies over rack center. Translate the base by
        # the same xy displacement as part-rack, subject to per-step bounds. Base-x
        # positive is often blocked, so vary x alignment across trials.
        dx, dy = float(part[0] - rack[0]), float(part[1] - rack[1])
        # First cover a regular local xy grid.  The nominal offset should place the
        # unchanged tool directly over the part if base/world axes are aligned.
        grid = (-0.04, -0.02, 0.0, 0.02, 0.04)
        target_dx = dx + grid[(trial // len(grid)) % len(grid)]
        target_dy = dy + grid[trial % len(grid)]
        for axis, total in ((0, target_dx), (1, target_dy)):
            while abs(total) > 1e-6:
                action = np.zeros(11)
                move = max(-0.2, min(0.2, total))
                action[axis] = move
                action[10] = 1.0  # approach open
                state, success = step_and_check(env, state, action, "align", axis)
                if success:
                    return True
                total -= move

        # Most important test: preserve the default arm pose and sustain a close at
        # the aligned location.  Also dither one arm coordinate at a time to sweep
        # through contact instead of jumping to an unrelated random posture.
        for step in range(40):
            action = np.zeros(11)
            if step >= 8:
                axis = 3 + ((trial // 25) % 7)
                direction = -1.0 if ((trial // 175) % 2) else 1.0
                action[axis] = direction * 0.012
            action[10] = -1.0
            state, success = step_and_check(env, state, action, "aligned-close", step)
            if success:
                return True

        # Search locally around the known default posture. Each trial executes a
        # smooth bounded random walk plus periodic close events. Large arm changes
        # are avoided because the default pose already puts the tool at table height.
        phase = rng.uniform(0.0, 2.0 * math.pi)
        for step in range(240):
            action = np.zeros(11)
            # Explore base-y around the alignment; base-x only toward its permitted
            # negative direction. Oscillations make this deterministic-ish and dense.
            action[1] = 0.025 * math.sin(phase + step * 0.45)
            if step % 31 == 0:
                action[0] = rng.choice((0.0, -0.02, -0.04))

            # Wrist/shoulder perturbations: mostly tiny, with occasional wider step.
            scale = 0.018 if step < 120 else 0.04
            for index in range(3, 10):
                action[index] = max(-0.08, min(0.08, rng.gauss(0.0, scale)))

            # Alternate open approach and close attempts; sustain closure for several
            # steps because a one-step command did not alter finger_state in isolation.
            cycle = step % 12
            action[10] = -1.0 if cycle >= 5 else 1.0
            state, success = step_and_check(env, state, action, "walk", step)
            if success:
                return True
        return False
    finally:
        env.close()


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    for trial in range(trials):
        rng = random.Random(1907 + trial)
        print("trial", trial, flush=True)
        if run_trial(trial, rng):
            return 0
    print("NO GRASP", trials, "trials")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

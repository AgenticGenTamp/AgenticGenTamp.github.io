"""Focused black-box probes for Packing3D action semantics.

This file is deliberately independent of approach.py.  Every experiment uses a
new server-side environment and a deterministic reset seed so its first-step
state can be compared with a no-op baseline.
"""

from __future__ import annotations

import numpy as np

from env_client import make_env


ROBOT_FEATURES = [
    "pos_base_x", "pos_base_y", "pos_base_rot",
    "joint_1", "joint_2", "joint_3", "joint_4",
    "joint_5", "joint_6", "joint_7",
    "finger_state", "grasp_active",
]


def robot_values(env, state):
    robot_type = env.observation_space.get_type("Kinematic3DRobot")
    robot = state.get_objects(robot_type)[0]
    return np.array([state.get(robot, f) for f in ROBOT_FEATURES])


def part_values(env, state):
    rows = {}
    for type_name in ("Kinematic3DCuboid", "Kinematic3DTriangle"):
        typ = env.observation_space.get_type(type_name)
        for obj in state.get_objects(typ):
            if obj.name.startswith("part"):
                rows[obj.name] = np.array(
                    [state.get(obj, f) for f in ("pose_x", "pose_y", "pose_z")]
                )
    return rows


def one_step(action, seed=17):
    env = make_env()
    try:
        state0, info0 = env.reset(seed=seed)
        robot0 = robot_values(env, state0)
        parts0 = part_values(env, state0)
        state1, reward, terminated, truncated, info1 = env.step(
            np.asarray(action, dtype=np.float32)
        )
        return {
            "robot0": robot0,
            "robot1": robot_values(env, state1),
            "parts0": parts0,
            "parts1": part_values(env, state1),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info0": info0,
            "info1": info1,
        }
    finally:
        env.close()


def action_scaling_probe():
    print("ACTION SPACE / METADATA")
    env = make_env()
    try:
        print(" shape", env.action_space.shape)
        print(" low", env.action_space.low.tolist())
        print(" high", env.action_space.high.tolist())
        print(" dtype", env.action_space.dtype)
        print(" max_steps", env.max_steps)
    finally:
        env.close()

    zero = np.zeros(11, dtype=np.float32)
    baseline = one_step(zero)
    print(" initial robot", dict(zip(ROBOT_FEATURES, baseline["robot0"].tolist())))
    base_delta = baseline["robot1"] - baseline["robot0"]
    print("\nNO-OP robot delta", dict(zip(ROBOT_FEATURES, base_delta.tolist())))
    for index in range(10):
        action = zero.copy()
        action[index] = 0.2
        result = one_step(action)
        # Subtract the deterministic one-step no-op drift as well as initial state.
        raw = result["robot1"] - result["robot0"]
        effect = raw - base_delta
        changed = {
            name: round(float(value), 7)
            for name, value in zip(ROBOT_FEATURES, effect)
            if abs(value) > 1e-7
        }
        print(f" action[{index}]=+0.2 effect-vs-noop", changed)

    # Check sign and approximate linearity on representative base/joint controls.
    for index in (0, 2, 3, 9):
        for magnitude in (-0.2, -0.1, 0.1):
            action = zero.copy()
            action[index] = magnitude
            result = one_step(action)
            effect = result["robot1"] - result["robot0"] - base_delta
            print(
                f" action[{index}]={magnitude:+.1f}",
                {name: round(float(value), 7) for name, value in
                 zip(ROBOT_FEATURES, effect) if abs(value) > 1e-7},
            )


def gripper_probe():
    print("\nGRIPPER THRESHOLDS")
    for command in (-1.0, -0.51, -0.5, 0.0, 0.5, 0.51, 1.0):
        action = np.zeros(11, dtype=np.float32)
        action[10] = command
        result = one_step(action)
        r0, r1 = result["robot0"], result["robot1"]
        print(
            f" command={command:+.2f}",
            f"finger {r0[10]:.6f}->{r1[10]:.6f}",
            f"grasp {r0[11]:.1f}->{r1[11]:.1f}",
        )

    env = make_env()
    try:
        state, _ = env.reset(seed=17)
        robot_type = env.observation_space.get_type("Kinematic3DRobot")
        robot = state.get_objects(robot_type)[0]
        for command in (1.0, -1.0):
            action = np.zeros(11, dtype=np.float32)
            action[10] = command
            trace = []
            for _ in range(12):
                state, _, _, _, _ = env.step(action)
                trace.append(
                    (state.get(robot, "finger_state"), state.get(robot, "grasp_active"))
                )
            print(f" held command={command:+.1f} (finger,grasp)", trace)
    finally:
        env.close()


def repeated_action_probe():
    print("\nREPEATED ACTION / LIMIT BEHAVIOR")
    env = make_env()
    try:
        state, _ = env.reset(seed=17)
        initial = robot_values(env, state)
        action = np.zeros(11, dtype=np.float32)
        action[0] = 0.2
        for step in range(1, 6):
            state, reward, terminated, truncated, _ = env.step(action)
            current = robot_values(env, state)
            print(f" base +0.2 step {step}: dx={current[0]-initial[0]:.6f}")
    finally:
        env.close()

    env = make_env()
    try:
        state, _ = env.reset(seed=17)
        action = np.zeros(11, dtype=np.float32)
        final = None
        for step in range(1, env.max_steps + 2):
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                final = (step, reward, terminated, truncated, info)
                break
        print(" no-op episode first done", final)
    finally:
        env.close()


if __name__ == "__main__":
    import sys

    requested = set(sys.argv[1:]) or {"actions", "gripper", "limits"}
    if "actions" in requested:
        action_scaling_probe()
    if "gripper" in requested:
        gripper_probe()
    if "limits" in requested:
        repeated_action_probe()

"""Scan candidate vertical grasp configurations and both gripper polarities."""

import math
import numpy as np

from env_client import make_env


POSES = (
    ("initial", (0.0, -0.3491, math.pi, -2.5482, 0.0, -0.8727, math.pi / 2)),
    ("p45", (0.0, 0.45, math.pi, -1.11, 0.0, -1.58, math.pi / 2)),
    ("p50", (0.0, 0.50, math.pi, -0.70, 0.0, -1.93, math.pi / 2)),
    ("p65", (0.0, 0.65, math.pi, -0.24, 0.0, -2.25, math.pi / 2)),
)


def val(state, obj, feat):
    return float(state.get(obj, feat))


def servo(env, state, target_base, target_q, grip, steps):
    robot = state.get_object_from_name("robot")
    for _ in range(steps):
        action = np.zeros(18, dtype=np.float32)
        current_base = np.array([val(state, robot, f) for f in
                                 ("pos_base_x", "pos_base_y", "pos_base_rot")])
        current_q = np.array([val(state, robot, f"pos_arm_joint{i}") for i in range(1, 8)])
        action[:3] = np.clip(np.asarray(target_base) - current_base, -0.1, 0.1)
        action[3:10] = np.clip(np.asarray(target_q) - current_q, -0.1, 0.1)
        action[10] = grip
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return state, reward, terminated, truncated


def xyz(state, obj):
    return np.array([val(state, obj, f) for f in ("x", "y", "z")])


def main():
    env = make_env()
    try:
        for label, pose in POSES:
            for open_value, close_value in ((0.0, 1.0), (1.0, 0.0)):
                state, _ = env.reset(seed=0, options={"object_count": 1})
                cube = state.get_object_from_name("cube_0")
                robot = state.get_object_from_name("robot")
                start = xyz(state, cube)
                target_base = (start[0] - 0.50, start[1], 0.0)

                state, reward, term, trunc = servo(
                    env, state, target_base, pose, open_value, 70)
                at_pose = xyz(state, cube)
                grip_pos = val(state, robot, "pos_gripper")
                state, reward, term, trunc = servo(
                    env, state, target_base, pose, close_value, 30)
                after_close = xyz(state, cube)
                closed_pos = val(state, robot, "pos_gripper")

                # Lift to the initial folded/high configuration before translating;
                # this avoids interpreting ground friction as a failed grasp.
                state, reward, term, trunc = servo(
                    env, state, target_base, POSES[0][1], close_value, 55)
                after_lift = xyz(state, cube)

                retreat = (target_base[0], target_base[1] - 0.40, 0.0)
                state, reward, term, trunc = servo(
                    env, state, retreat, POSES[0][1], close_value, 45)
                final = xyz(state, cube)
                base_final = np.array([val(state, robot, "pos_base_x"),
                                       val(state, robot, "pos_base_y")])
                print(label, "open/close", open_value, close_value,
                      "gripper", round(grip_pos, 3), round(closed_pos, 3),
                      "start", np.round(start, 3).tolist(),
                      "at", np.round(at_pose, 3).tolist(),
                      "closed", np.round(after_close, 3).tolist(),
                      "lift", np.round(after_lift, 3).tolist(),
                      "final", np.round(final, 3).tolist(),
                      "delta", np.round(final - start, 3).tolist(),
                      "base", np.round(base_final, 3).tolist(),
                      "reward", reward, "done", term, trunc,
                      flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

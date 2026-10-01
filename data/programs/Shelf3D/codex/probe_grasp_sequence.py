"""Test the best rendered arm-only grasp and lift sequence."""

import numpy as np

from env_client import make_env


LOW = np.array([-0.136, 1.898, 3.134, -0.720, 0.014, -0.523, 1.571])
HIGH = np.array([-0.117, 1.078, 3.140, -0.907, 0.0, -1.157, 1.571])


def value(state, name, feature):
    return state.get(state.get_object_from_name(name), feature)


def pose(state, cube):
    return np.array([value(state, cube, f) for f in ("x", "y", "z")])


def drive(env, state, target, grip, limit=160):
    for _ in range(limit):
        q = np.array([value(state, "robot", f"pos_arm_joint{i}") for i in range(1, 8)])
        error = target - q
        if np.max(np.abs(error)) < 0.025:
            break
        action = np.zeros(11, dtype=np.float32)
        action[3:10] = np.clip(0.15 * error, -0.1, 0.1)
        action[10] = grip
        state, *_ = env.step(action)
    for _ in range(5):
        action = np.zeros(11, dtype=np.float32)
        action[10] = grip
        state, *_ = env.step(action)
    return state


def main():
    for close_value in (1.0, 0.0):
        env = make_env()
        state, _ = env.reset(seed=0, options={"object_count": 1})
        cube = next(n for n in state.get_object_names() if n.startswith("cube"))
        start = pose(state, cube)
        # Bring the cube 10 cm closer to the arm so a lower vertical posture
        # remains within reach, while leaving it beyond the chassis edge.
        for command in (0.1, 0.015, 0.0):
            action = np.zeros(11, dtype=np.float32)
            action[0] = command
            state, *_ = env.step(action)
        state = drive(env, state, LOW, 0.0)
        low = pose(state, cube)
        if close_value == 1.0:
            print("low render", env.render_state(state=state, label="best_grasp_low"))
        for _ in range(10):
            action = np.zeros(11, dtype=np.float32)
            action[10] = close_value
            state, *_ = env.step(action)
        closed = pose(state, cube)
        state = drive(env, state, HIGH, close_value)
        lifted = pose(state, cube)
        print("grip", close_value, "cube", *(np.round(p, 4).tolist()
              for p in (start, low, closed, lifted)),
              "render", env.render_state(state=state, label=f"lift_grip_{close_value:g}"))
        env.close()


if __name__ == "__main__":
    main()

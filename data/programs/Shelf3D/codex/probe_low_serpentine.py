"""Serpentine contact scan for the requested very-low arm posture."""

import numpy as np

from env_client import make_env


def get(state, name, feature):
    return state.get(state.get_object_from_name(name), feature)


def cube_pose(state, cube):
    return np.array([get(state, cube, f) for f in ("x", "y", "z")])


def drive_arm(env, state, target):
    for _ in range(260):
        q = np.array([get(state, "robot", f"pos_arm_joint{i}") for i in range(1, 8)])
        error = target - q
        if np.max(np.abs(error)) < 0.03:
            break
        action = np.zeros(11, dtype=np.float32)
        action[3:10] = np.clip(0.25 * error, -0.1, 0.1)
        state, *_ = env.step(action)
    return state


def drive_base(env, state, target_xy):
    for _ in range(25):
        here = np.array([get(state, "robot", f) for f in ("pos_base_x", "pos_base_y")])
        error = target_xy - here
        if np.max(np.abs(error)) < 0.008:
            break
        action = np.zeros(11, dtype=np.float32)
        action[:2] = np.clip(error / 0.87, -0.1, 0.1)
        state, *_ = env.step(action)
    return state


def trial(q5):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 1})
    cube = next(n for n in state.get_object_names() if n.startswith("cube"))
    cube0 = cube_pose(state, cube)
    base0 = np.array([get(state, "robot", f) for f in ("pos_base_x", "pos_base_y")])
    target = np.array([0.0, 2.4, 0.0, -2.55, q5, -2.1, 1.571])
    state = drive_arm(env, state, target)
    actual = np.array([get(state, "robot", f"pos_arm_joint{i}") for i in range(1, 8)])
    first = None
    # Increase x toward the cube; alternate lateral sweep direction each row.
    y_offsets = np.linspace(-0.6, 0.6, 13)
    for row, xoff in enumerate(np.linspace(0.0, 0.5, 6)):
        ys = y_offsets if row % 2 == 0 else y_offsets[::-1]
        for yoff in ys:
            state = drive_base(env, state, base0 + [xoff, yoff])
            moved = cube_pose(state, cube) - cube0
            if np.linalg.norm(moved) > 0.005:
                first = (float(xoff), float(yoff), moved.copy())
                break
        if first is not None:
            break
    final = cube_pose(state, cube)
    print("q5", q5, "actual", np.round(actual, 3).tolist(),
          "first", None if first is None else (round(first[0], 3), round(first[1], 3),
          np.round(first[2], 4).tolist()), "cube", np.round(cube0, 4).tolist(),
          "->", np.round(final, 4).tolist())
    env.close()


if __name__ == "__main__":
    for q5 in (2.0, 1.0, 3.0):
        trial(q5)

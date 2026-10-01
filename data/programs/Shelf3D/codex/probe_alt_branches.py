"""Settle and grasp-test low alternate FK branches at the joint-2 limit."""

import numpy as np

from env_client import make_env
from solve_alt_fk import fk


def get(state, name, feature):
    return state.get(state.get_object_from_name(name), feature)


def qpos(state):
    return np.array([get(state, "robot", f"pos_arm_joint{i}") for i in range(1, 8)])


def cube_pos(state, cube):
    return np.array([get(state, cube, f) for f in ("x", "y", "z")])


def drive(env, state, q_target=None, base_target=None, grip=0.0, steps=260):
    for _ in range(steps):
        action = np.zeros(11, dtype=np.float32)
        errors = []
        if q_target is not None:
            error = q_target - qpos(state)
            action[3:10] = np.clip(0.35 * error, -0.1, 0.1)
            errors.append(np.max(np.abs(error)))
        if base_target is not None:
            base = np.array([get(state, "robot", f) for f in ("pos_base_x", "pos_base_y")])
            error = base_target - base
            action[:2] = np.clip(error / 0.87, -0.1, 0.1)
            errors.append(np.max(np.abs(error)))
        action[10] = grip
        if errors and max(errors) < 0.025:
            break
        state, *_ = env.step(action)
    return state


def trial(label, target, align_offset=(0.0, 0.0)):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 1})
    cube = next(n for n in state.get_object_names() if n.startswith("cube"))
    cube0 = cube_pos(state, cube)
    initial_base = np.array([get(state, "robot", f)
                             for f in ("pos_base_x", "pos_base_y")])
    # Retreat before lowering so the long fingers do not nudge the cube while
    # the alternate branch settles; approach only after measuring actual FK.
    state = drive(env, state, base_target=initial_base + [-0.15, 0.0], steps=40)
    state = drive(env, state, q_target=target,
                  base_target=initial_base + [-0.15, 0.0])
    actual = qpos(state)
    transform = fk(actual)
    point, axis = transform[:3, 3], transform[:3, 2]
    # Calibrated mapping: arm-FK -x is robot world +x; y retains sign.
    base_target = (cube0[:2] - np.array([-point[0], point[1]])
                   + np.asarray(align_offset))
    state = drive(env, state, q_target=target, base_target=base_target)
    aligned = cube_pos(state, cube)
    if label.startswith("q4m10"):
        print("render", env.render_state(state=state, label=f"alt_branch_{label}"))
    for _ in range(10):
        action = np.zeros(11, dtype=np.float32)
        action[10] = 1.0
        state, *_ = env.step(action)
    closed = cube_pos(state, cube)
    # Near-vertical IK lift for the refined q4=-1, q6=-0.3 branch; this
    # preserves fingertip x/y instead of swinging the closed hand through cube.
    if abs(target[3] + 1.0) < 0.05 and abs(target[5] + 0.3) < 0.08:
        lift = np.array([0.0, 1.40, 3.139, -1.338, 0.005, -0.404, 1.572])
    else:
        lift = target.copy()
        lift[1] = 1.45
    state = drive(env, state, q_target=lift, grip=1.0)
    final = cube_pos(state, cube)
    print(label, "target", np.round(target, 3).tolist(),
          "actual", np.round(actual, 3).tolist(),
          "fk_p", np.round(point, 3).tolist(),
          "axis", np.round(axis, 3).tolist(),
          "base_target", np.round(base_target, 3).tolist(),
          "cube", *(np.round(v, 4).tolist() for v in (cube0, aligned, closed, final)))
    env.close()


if __name__ == "__main__":
    branches = [
        ("q4m06", np.array([0.0, 2.24, 3.14, -0.60, 3.14, 0.30, 1.571])),
        ("q4m10", np.array([0.0, 2.24, 3.14, -1.00, 0.00, 0.10, 1.571])),
        ("q4m12", np.array([0.0, 2.24, 3.14, -1.20, 3.14, -0.30, 1.571])),
        # Nearby non-singular q3/q5 branch from the grid ranking.
        ("nonsingular", np.array([0.0, 2.24, 2.945, -1.00, -0.982, 0.20, 1.571])),
    ]
    for name, q in branches:
        trial(name, q)

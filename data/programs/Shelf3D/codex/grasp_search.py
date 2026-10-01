"""Empirically search simple planar grasp postures on a one-cube episode."""

import numpy as np

from env_client import make_env


def get(state, name, feature):
    return state.get(state.get_object_from_name(name), feature)


def cube_name(state):
    return next(n for n in state.get_object_names() if n.startswith("cube"))


def step(env, action):
    return env.step(np.asarray(action, dtype=np.float32))[0]


def move_base(env, state, rel_x, rel_y=0.0, limit=30):
    cube = cube_name(state)
    for _ in range(limit):
        ex = get(state, cube, "x") - get(state, "robot", "pos_base_x") - rel_x
        ey = get(state, cube, "y") - get(state, "robot", "pos_base_y") - rel_y
        if abs(ex) < 0.008 and abs(ey) < 0.008:
            break
        action = np.zeros(11)
        action[0] = np.clip(ex / 0.87, -0.1, 0.1)
        action[1] = np.clip(ey / 0.87, -0.1, 0.1)
        state = step(env, action)
    for _ in range(2):
        state = step(env, np.zeros(11))
    return state


def move_arm(env, state, q_target, limit=100, grip=0.0):
    """Velocity-like feedback; low gain avoids latent-controller overshoot."""
    for _ in range(limit):
        q = np.array([get(state, "robot", f"pos_arm_joint{i}") for i in range(1, 8)])
        error = q_target - q
        if np.max(np.abs(error)) < 0.025:
            break
        action = np.zeros(11)
        action[3:10] = np.clip(0.12 * error, -0.1, 0.1)
        action[10] = grip
        state = step(env, action)
    for _ in range(5):
        action = np.zeros(11)
        action[10] = grip
        state = step(env, action)
    return state


def trial(rel_x, q2, q6, render=False):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 1})
    cube = cube_name(state)
    initial = np.array([get(state, cube, f) for f in ("x", "y", "z")])
    q_home = np.array([get(state, "robot", f"pos_arm_joint{i}") for i in range(1, 8)])
    state = move_base(env, state, rel_x)
    if render:
        print("render aligned", env.render_state(state=state, label="grasp_aligned"))
    aligned = np.array([get(state, cube, f) for f in ("x", "y", "z")])
    q_down = q_home.copy()
    q_down[1], q_down[5] = q2, q6
    state = move_arm(env, state, q_down, grip=0.0)
    if render:
        print("render down", env.render_state(state=state, label="grasp_down"))
    down = np.array([get(state, cube, f) for f in ("x", "y", "z")])
    # Close, then lift back toward the collision-safe home posture.
    for _ in range(8):
        action = np.zeros(11)
        action[10] = 1.0
        state = step(env, action)
    if render:
        print("render closed", env.render_state(state=state, label="grasp_closed"))
    closed = np.array([get(state, cube, f) for f in ("x", "y", "z")])
    state = move_arm(env, state, q_home, limit=100, grip=1.0)
    lifted = np.array([get(state, cube, f) for f in ("x", "y", "z")])
    q_final = np.array([get(state, "robot", f"pos_arm_joint{i}") for i in range(1, 8)])
    env.close()
    print("trial", rel_x, q2, q6,
          "cube", *(np.round(x, 3).tolist() for x in (initial, aligned, down, closed, lifted)),
          "qfinal", np.round(q_final, 2).tolist())
    return lifted[2]


if __name__ == "__main__":
    # Planar postures predicted to place the fingers near ground level.
    for args in [
        (0.10, -2.15, -1.56),
        (0.15, -2.16, -1.65),
        (0.20, -2.18, -1.75),
        (0.25, -2.20, -1.85),
        (0.30, -2.20, -1.95),
    ]:
        trial(*args, render=(args[0] == 0.20))

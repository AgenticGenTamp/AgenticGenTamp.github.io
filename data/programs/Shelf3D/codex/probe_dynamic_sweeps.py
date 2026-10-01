"""Search dynamic arm sweeps for cube contact on seed-0 one-cube layout."""

import math
import numpy as np

from env_client import make_env


def get(state, name, feature):
    return state.get(state.get_object_from_name(name), feature)


def cube_pose(state, cube):
    return np.array([get(state, cube, f) for f in ("x", "y", "z")])


def joints(state):
    return np.array([get(state, "robot", f"pos_arm_joint{i}") for i in range(1, 8)])


def advance_to(env, state, q_target, base_dx):
    base_start = np.array([get(state, "robot", f) for f in ("pos_base_x", "pos_base_y")])
    for _ in range(220):
        q = joints(state)
        base = np.array([get(state, "robot", f) for f in ("pos_base_x", "pos_base_y")])
        action = np.zeros(11, dtype=np.float32)
        action[:2] = np.clip((base_start + [base_dx, 0] - base) / 0.87, -0.1, 0.1)
        action[3:10] = np.clip(0.3 * (q_target - q), -0.1, 0.1)
        if max(np.max(np.abs(q_target - q)), abs(base_start[0] + base_dx - base[0])) < 0.04:
            break
        state, *_ = env.step(action)
    return state


def run(label, base_dx, center, amplitudes, periods, phases, steps=400):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 1})
    cube = next(n for n in state.get_object_names() if n.startswith("cube"))
    cube0 = cube_pose(state, cube)
    state = advance_to(env, state, center, base_dx)
    start_q = joints(state)
    first = None
    for t in range(steps):
        desired = center + amplitudes * np.sin(
            2 * math.pi * t / periods + phases)
        q = joints(state)
        action = np.zeros(11, dtype=np.float32)
        action[3:10] = np.clip(0.45 * (desired - q), -0.1, 0.1)
        # Exercise both open sweeping contacts and intermittent closures.
        action[10] = 1.0 if (t // 25) % 2 else 0.0
        state, *_ = env.step(action)
        displacement = cube_pose(state, cube) - cube0
        if np.linalg.norm(displacement) > 0.003:
            first = {
                "step": t,
                "q": np.round(joints(state), 4).tolist(),
                "action": np.round(action, 4).tolist(),
                "disp": np.round(displacement, 4).tolist(),
            }
            break
    print(label, "base_dx", base_dx, "start_q", np.round(start_q, 3).tolist(),
          "first", first, "final_cube", np.round(cube_pose(state, cube), 4).tolist())
    env.close()
    return first


def main():
    pi = math.pi
    specs = [
        # Vertical fingers over the edge: flex shoulder/elbow/wrist out of phase.
        ("vertical_edge", 0.15,
         [-0.14, 2.05, 3.14, -0.55, 0.0, -0.50, 1.571],
         [0.20, 0.20, 0.0, 0.50, 0.35, 0.45, 0.0],
         [100, 130, 1, 150, 80, 150, 1],
         [0, 0, 0, pi, pi / 2, 0, 0]),
        # Folded configuration requested previously; sweep wrist axes strongly.
        ("folded_edge", 0.15,
         [0.0, 2.20, 0.0, -2.40, 2.0, -1.90, 1.571],
         [0.40, 0.18, 0.45, 0.15, 1.0, 0.20, 0.8],
         [140, 110, 170, 90, 120, 100, 130],
         [0, pi / 2, pi / 2, pi, 0, 0, pi / 2]),
        # Close base alignment and asymmetric wrist swing around lowest IK family.
        ("close_edge", 0.28,
         [-0.20, 2.18, 3.14, -0.60, 0.0, -0.30, 1.571],
         [0.18, 0.18, 0.0, 0.50, 0.8, 0.55, 0.8],
         [90, 120, 1, 140, 100, 140, 100],
         [0, 0, 0, pi, pi / 2, 0, 0]),
    ]
    for spec in specs:
        converted = [spec[0], spec[1]] + [np.asarray(x, dtype=float) for x in spec[2:]]
        result = run(*converted)
        if result is not None:
            break


if __name__ == "__main__":
    main()

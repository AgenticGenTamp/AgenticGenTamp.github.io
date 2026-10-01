"""Deterministic local search for a genuine close-and-lift configuration.

This is an exploration script only; approach.py does not import it.
"""
import sys

import numpy as np

from env_client import make_env


def read(state, name, features):
    obj = state.get_object_from_name(name)
    return np.array([float(state.get(obj, f)) for f in features])


def robot(state):
    return read(
        state,
        "robot",
        ["pos_base_x", "pos_base_y", "pos_base_rot"]
        + [f"pos_arm_joint{i}" for i in range(1, 8)]
        + ["pos_gripper"],
    )


def cube_pos(state, names):
    return {n: read(state, n, ("x", "y", "z")) for n in names}


def command(state, base, joints, grip, base_limit=0.1):
    r = robot(state)
    error = np.asarray(base) - r[:3]
    error[2] = (error[2] + np.pi) % (2 * np.pi) - np.pi
    action = np.zeros(11, dtype=np.float32)
    action[:3] = np.clip(0.8 * error, -base_limit, base_limit)
    action[3:10] = np.clip(0.7 * (np.asarray(joints) - r[3:10]), -0.1, 0.1)
    action[10] = grip
    return action


def run(env, state, count, base, joints, grip, peaks, names, base_limit=0.1):
    for _ in range(count):
        state, _, _, _, _ = env.step(
            command(state, base, joints, grip, base_limit)
        )
        for n, p in cube_pos(state, names).items():
            peaks[n] = max(peaks[n], p[2])
    return state


def parameters(index):
    # Low-discrepancy-like deterministic coverage, concentrated around the
    # visually verified downward-facing pose.
    rng = np.random.default_rng(1729 + index)
    if index >= 100:
        # Local refinement around trial 14, the only broad sample where open
        # descent and jaw closure both produced substantial object motion.
        center = np.array(
            [-0.0558, 1.2931, 3.0625, -1.5832, 0.0122, 0.6920, 1.5211]
        )
        scale = np.array([0.08, 0.10, 0.08, 0.14, 0.10, 0.18, 0.16])
        q = center + rng.uniform(-1.0, 1.0, 7) * scale
        return q, rng.uniform(-0.025, 0.025), 0.0401 + rng.uniform(-0.025, 0.025)
    q = np.array(
        [
            rng.uniform(-0.32, 0.32),
            rng.uniform(1.16, 1.48),
            np.pi + rng.uniform(-0.25, 0.25),
            rng.uniform(-1.85, -1.15),
            rng.uniform(-0.35, 0.35),
            rng.uniform(0.55, 1.45),
            np.pi / 2 + rng.uniform(-0.45, 0.45),
        ]
    )
    return q, rng.uniform(-0.07, 0.07), rng.uniform(-0.045, 0.045)


def trial(index):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 4})
    names = sorted(n for n in state.get_object_names() if n.startswith("cube"))
    initial = cube_pos(state, names)
    # Prefer the highest cube, which is least occluded by the pile.
    target = max(names, key=lambda n: initial[n][2])
    qlow, dx, dy = parameters(index)
    p = initial[target]
    base = np.array([p[0] + 0.91 + dx, p[1] + dy, np.pi])
    staging = np.array([1.0, base[1], np.pi])
    # A shoulder reversal is the best-established approximate vertical lift.
    qhigh = qlow.copy()
    qhigh[1] = max(0.72, qlow[1] - 0.46)
    peaks = {n: initial[n][2] for n in names}
    state = run(env, state, 80, staging, qhigh, 1.0, peaks, names)
    state = run(env, state, 30, base, qhigh, 1.0, peaks, names)
    before = cube_pos(state, names)
    state = run(env, state, 45, base, qlow, 1.0, peaks, names)
    descended = cube_pos(state, names)
    state = run(env, state, 24, base, qlow, 0.0, peaks, names)
    closed = cube_pos(state, names)
    state = run(env, state, 55, base, qhigh, 0.0, peaks, names)
    final = cube_pos(state, names)
    env.close()
    max_contact = max(np.linalg.norm(descended[n] - before[n]) for n in names)
    max_close = max(np.linalg.norm(closed[n] - descended[n]) for n in names)
    best = max(names, key=lambda n: peaks[n] - initial[n][2])
    rise = peaks[best] - initial[best][2]
    retained = final[best][2] - initial[best][2]
    print(
        "IDX", index, "target", target,
        "q", np.round(qlow, 4).tolist(),
        "dxdy", [round(dx, 4), round(dy, 4)],
        "contact", round(max_contact, 4), "close", round(max_close, 4),
        "best", best, "rise", round(rise, 4), "retained", round(retained, 4),
        "final", np.round(final[best], 4).tolist(),
    )


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    stop = int(sys.argv[2]) if len(sys.argv) > 2 else start + 1
    for trial_index in range(start, stop):
        trial(trial_index)

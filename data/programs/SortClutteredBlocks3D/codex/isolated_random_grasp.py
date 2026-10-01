"""Randomized close/lift trials after sweeping cube1 clear of the pile."""
import sys

import numpy as np

from env_client import make_env
from topdown_grasp_probe import act, rob, run, xyz


EDGE = np.array([0.0, 1.3, np.pi, -1.7, 0.0, 1.0, 0.0])


def trial(index):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 4})
    home = rob(state)[3:10]
    cubes = sorted(n for n in state.get_object_names() if n.startswith("cube"))
    target = "cube1"
    # Reproducibly sweep cube1 away in -y with a narrow fingertip.
    state = run(env, state, 45, [1, 0.8, np.pi], home, 1)
    state = run(env, state, 55, [-1, 0.8, np.pi], home, 1)
    state = run(env, state, 45, [-1, 0.8, 0], home, 1)
    y = float(xyz(state, target)[1] + 0.042)
    state = run(env, state, 45, [-1, y, 0], home, 1)
    state = run(env, state, 130, [-1, y, 0], EDGE, 1)
    old = {n: xyz(state, n).copy() for n in cubes}
    hit = None
    for _ in range(40):
        action = act(state, [-0.8, y, 0], EDGE, 1)
        action[0] = min(action[0], 0.012)
        state, _, _, _, _ = env.step(action)
        if max(np.linalg.norm(xyz(state, n) - old[n]) for n in cubes) > 0.001:
            hit = rob(state)[:3].copy()
            break
    if hit is None:
        env.close()
        print("IDX", index, "isolation_failed")
        return
    sweep = EDGE.copy()
    sweep[0] = 0.5
    for _ in range(80):
        state, _, _, _, _ = env.step(act(state, hit, sweep, 1))
        if xyz(state, target)[1] < -0.085:
            break
    isolated = xyz(state, target).copy()

    rng = np.random.default_rng(9107 + index)
    low = np.array(
        [
            rng.uniform(-0.18, 0.18),
            rng.uniform(1.20, 1.43),
            np.pi + rng.uniform(-0.16, 0.16),
            rng.uniform(-1.82, -1.35),
            rng.uniform(-0.18, 0.18),
            rng.uniform(0.65, 1.30),
            np.pi / 2 + rng.uniform(-0.28, 0.28),
        ]
    )
    high = low.copy()
    high[1] = max(0.80, low[1] - 0.42)
    bx = isolated[0] - 0.91 + rng.uniform(-0.045, 0.045)
    by = isolated[1] + rng.uniform(-0.045, 0.045)
    state = run(env, state, 30, [-1.08, rob(state)[1], 0], sweep, 1)
    state = run(env, state, 100, [-1.08, by, 0], high, 1)
    state = run(env, state, 35, [bx, by, 0], high, 1)
    before = xyz(state, target).copy()
    state = run(env, state, 50, [bx, by, 0], low, 1)
    down = xyz(state, target).copy()
    state = run(env, state, 28, [bx, by, 0], low, 0)
    closed = xyz(state, target).copy()
    peak = closed[2]
    for _ in range(55):
        state, _, _, _, _ = env.step(act(state, [bx, by, 0], high, 0))
        peak = max(peak, xyz(state, target)[2])
    final = xyz(state, target).copy()
    env.close()
    print(
        "IDX", index, "q", np.round(low, 4).tolist(),
        "base", [round(bx, 4), round(by, 4)],
        "isolated", np.round(isolated, 4).tolist(),
        "down", np.round(down - before, 4).tolist(),
        "close", np.round(closed - down, 4).tolist(),
        "rise", round(peak - isolated[2], 4),
        "retained", round(final[2] - isolated[2], 4),
        "final", np.round(final, 4).tolist(),
    )


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    stop = int(sys.argv[2]) if len(sys.argv) > 2 else start + 1
    for i in range(start, stop):
        trial(i)

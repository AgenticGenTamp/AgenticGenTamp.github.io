"""Narrow top-down grasp sweep after first isolating cube1 with a q1 rake."""

from env_client import make_env
import numpy as np

from topdown_grasp_probe import act, rob, run, xyz


EDGE = np.array([0.0, 1.3, np.pi, -1.7, 0.0, 1.0, 0.0])
HIGH = np.array([0.0, 0.9, np.pi, -1.7, 0.0, 1.0, np.pi / 2])

# A balanced 3x3 sweep: every q2, y offset, q6, and q7 value occurs 3 times.
TRIALS = [
    (1.21, -0.005, 0.85, np.pi / 4 - 0.15),
    (1.21, 0.000, 1.00, np.pi / 4),
    (1.21, 0.005, 1.15, np.pi / 4 + 0.15),
    (1.22, -0.005, 1.00, np.pi / 4 + 0.15),
    (1.22, 0.000, 1.15, np.pi / 4 - 0.15),
    (1.22, 0.005, 0.85, np.pi / 4),
    (1.23, -0.005, 1.15, np.pi / 4),
    (1.23, 0.000, 0.85, np.pi / 4 + 0.15),
    (1.23, 0.005, 1.00, np.pi / 4 - 0.15),
]


def trial(params):
    low_q2, yoff, q6, q7 = params
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 4})
    home = rob(state)[3:10]
    cube = "cube1"
    cubes = [n for n in state.get_object_names() if n.startswith("cube")]

    # Reach the pile from the left, then use the known tangential sweep to
    # separate cube1 toward -y before attempting a grasp.
    state = run(env, state, 45, [1, 0.8, np.pi], home, 1)
    state = run(env, state, 55, [-1, 0.8, np.pi], home, 1)
    state = run(env, state, 45, [-1, 0.8, 0], home, 1)
    rake_y = float(xyz(state, cube)[1] + 0.042)
    state = run(env, state, 45, [-1, rake_y, 0], home, 1)
    state = run(env, state, 130, [-1, rake_y, 0], EDGE, 1)
    prior = {n: xyz(state, n).copy() for n in cubes}
    hit_base = None
    for _ in range(40):
        action = act(state, [-0.8, rake_y, 0], EDGE, 1)
        action[0] = min(action[0], 0.012)
        state, *_ = env.step(action)
        if max(np.linalg.norm(xyz(state, n) - prior[n]) for n in cubes) > 0.001:
            hit_base = rob(state)[:3].copy()
            break
    if hit_base is None:
        env.close()
        return {"params": params, "error": "no rake contact"}
    swept = EDGE.copy()
    swept[0] = 0.5
    for _ in range(80):
        state, *_ = env.step(act(state, hit_base, swept, 1))
        if xyz(state, cube)[1] < -0.085:
            break

    isolated = xyz(state, cube).copy()
    state = run(env, state, 25, [-1.08, rob(state)[1], 0], swept, 1)
    bx = float(xyz(state, cube)[0] - 0.91)
    by = float(xyz(state, cube)[1] + yoff)
    state = run(env, state, 100, [-1.08, by, 0], HIGH, 1)
    state = run(env, state, 30, [bx, by, 0], HIGH, 1)

    low = HIGH.copy()
    low[1] = low_q2
    low[5] = q6
    low[6] = q7
    before = xyz(state, cube).copy()
    state = run(env, state, 60, [bx, by, 0], low, 1)
    down = xyz(state, cube).copy()
    state = run(env, state, 20, [bx, by, 0], low, 0)
    closed = xyz(state, cube).copy()

    max_z = float(closed[2])
    max_follow = 0.0
    for _ in range(70):
        state, *_ = env.step(act(state, [bx, by, 0], HIGH, 0))
        now = xyz(state, cube)
        max_z = max(max_z, float(now[2]))
        max_follow = max(max_follow, float(np.linalg.norm(now[:2] - closed[:2])))
    after = xyz(state, cube).copy()
    result = {
        "params": tuple(round(float(v), 5) for v in params),
        "isolated": np.round(isolated, 5).tolist(),
        "before": np.round(before, 5).tolist(),
        "down": np.round(down, 5).tolist(),
        "closed": np.round(closed, 5).tolist(),
        "after": np.round(after, 5).tolist(),
        "max_z": round(max_z, 5),
        "max_follow_xy": round(max_follow, 5),
    }
    env.close()
    return result


if __name__ == "__main__":
    for index, parameters in enumerate(TRIALS):
        print("RESULT", index, trial(parameters), flush=True)

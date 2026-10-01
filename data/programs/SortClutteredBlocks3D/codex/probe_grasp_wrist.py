"""Bounded wrist/base grid around the visually aligned grasp family."""

import itertools

import numpy as np

from env_client import make_env


def xyz(state, name):
    obj = state.get_object_from_name(name)
    return np.array([state.get(obj, f) for f in ("x", "y", "z")])


def command(env, state, base_goal, q_goal, grip, steps):
    peak = {n: xyz(state, n)[2] for n in state.get_object_names()
            if n.startswith("cube")}
    for _ in range(steps):
        robot = state.get_object_from_name("robot")
        base = np.array([state.get(robot, f) for f in
                         ("pos_base_x", "pos_base_y", "pos_base_rot")])
        q = np.array([state.get(robot, "pos_arm_joint%d" % i)
                      for i in range(1, 8)])
        action = np.zeros(11, dtype=np.float32)
        error = base_goal - base
        error[2] = (error[2] + np.pi) % (2 * np.pi) - np.pi
        action[:3] = np.clip(1.5 * error, -0.1, 0.1)
        action[3:10] = np.clip(1.5 * (q_goal - q), -0.1, 0.1)
        action[10] = grip
        state, reward, terminated, truncated, _ = env.step(action)
        for name in peak:
            peak[name] = max(peak[name], xyz(state, name)[2])
        if terminated or truncated:
            break
    return state, peak


def trial(number, gap, q5, q6, q7):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 4})
    names = sorted(n for n in state.get_object_names() if n.startswith("cube"))
    initial = {n: xyz(state, n) for n in names}
    target = initial["cube3"]
    base_goal = np.array([target[0] + gap, target[1], np.pi])
    high = np.array([0.0, 1.00, np.pi, -1.00, q5, q6, q7])
    low = high.copy(); low[1] = 1.16
    lift = low.copy(); lift[1] = 0.82

    # Open and move above/behind, descend in place, close, then lift.
    state, p1 = command(env, state, base_goal, high, 1.0, 35)
    state, p2 = command(env, state, base_goal, low, 1.0, 20)
    state, p3 = command(env, state, base_goal, low, 0.0, 10)
    state, p4 = command(env, state, base_goal, lift, 0.0, 30)
    peak = {n: max(p1[n], p2[n], p3[n], p4[n]) for n in names}
    final = {n: xyz(state, n) for n in names}
    moved = sorted(names, key=lambda n: np.linalg.norm(final[n] - initial[n]),
                   reverse=True)
    best = moved[0]
    print(number, "gap", gap, "wrist", (q5, q6, q7),
          "peak3", round(peak["cube3"] - initial["cube3"][2], 3),
          "end3", np.round(final["cube3"] - initial["cube3"], 3),
          "maxmove", best, np.round(final[best] - initial[best], 3))
    env.close()
    return peak["cube3"] - initial["cube3"][2]


def main():
    configs = list(itertools.product((0.42, 0.46, 0.50), (0.7, 1.0, 1.3),
                                     (0.0, np.pi / 2)))
    # At the calibrated middle stand-off, also rotate joint 5 both ways.
    configs += [(0.46, q6, q7, q5)
                for q5 in (-1.0, 1.0)
                for q6 in (0.7, 1.0)
                for q7 in (0.0, np.pi / 2)]
    normalized = []
    for config in configs:
        if len(config) == 3:
            gap, q6, q7 = config; q5 = 0.0
        else:
            gap, q6, q7, q5 = config
        normalized.append((gap, q5, q6, q7))
    results = []
    for i, config in enumerate(normalized, 1):
        results.append((trial(i, *config), config))
    print("BEST", max(results))


if __name__ == "__main__":
    main()

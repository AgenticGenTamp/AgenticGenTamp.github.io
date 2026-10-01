"""Small reproducible macro search on seed 0 / four objects.

Each genome describes a west-side edge-rake maneuver.  A candidate is always
evaluated in a fresh environment.  This is deliberately tiny (two generations
of four candidates) so it remains a diagnostic rather than an online planner.
"""

import math

import numpy as np

from env_client import make_env


COLORS = ("red", "green", "blue", "yellow")
FOLD = np.array([0.0, 0.0, math.pi, 0.0, 0.0, 0.0, 0.0])


def values(state, name, features):
    obj = state.get_object_from_name(name)
    return np.array([float(state.get(obj, f)) for f in features])


def robot(state):
    return values(state, "robot", ["pos_base_x", "pos_base_y", "pos_base_rot"] +
                  [f"pos_arm_joint{i}" for i in range(1, 8)] + ["pos_gripper"])


def command(state, base, joints, grip=0.0, limit=0.1):
    now = robot(state)
    action = np.zeros(11, dtype=np.float32)
    if base is not None:
        action[:3] = np.clip(0.8 * (np.asarray(base) - now[:3]), -limit, limit)
    if joints is not None:
        action[3:10] = np.clip(0.8 * (np.asarray(joints) - now[3:10]), -.1, .1)
    action[10] = grip
    return action


def hold(env, state, base, joints, steps, limit=.1):
    reward = -1.0
    for _ in range(steps):
        state, reward, term, trunc, _ = env.step(
            command(state, base, joints, limit=limit))
        if term or trunc:
            break
    return state, float(reward), bool(term or trunc)


def score(state, cube_colors):
    distances = {}
    for cube, color in cube_colors.items():
        cxy = values(state, cube, ("x", "y"))
        bxy = values(state, "bin_" + color, ("x", "y"))
        distances[cube] = float(np.linalg.norm(cxy - bxy))
    vals = np.array(list(distances.values()))
    # Max distance is the primary minimax objective; sum breaks ties.
    return float(vals.max() + .25 * vals.sum()), distances


def evaluate(genome):
    # genome = lateral station, depth, q1 sweep, q2, q4, q6, q7, retreat dx
    y, depth, sweep, q2, q4, q6, q7, retreat = genome
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 4})
    names = sorted(n for n in state.get_object_names() if n.startswith("cube"))
    offset = min(int(n[4:]) for n in names)
    cube_colors = {n: COLORS[(int(n[4:]) - offset) % 4] for n in names}
    initial_objective, initial_distances = score(state, cube_colors)
    edge = np.array([0.0, q2, math.pi, q4, 0.0, q6, q7])

    # Safe exterior route, followed by a slow final contact approach.
    macros = [
        ([1.00, .82, math.pi], FOLD, 35, .1),
        ([-1.00, .82, math.pi], FOLD, 45, .1),
        ([-1.00, y, 0.0], FOLD, 40, .1),
        ([-1.00, y, 0.0], edge, 75, .1),
        ([depth, y, 0.0], edge, 28, .012),
    ]
    reward = -1.0
    done = False
    for base, joints, steps, limit in macros:
        state, reward, done = hold(env, state, base, joints, steps, limit)
        if done:
            break
    if not done:
        swept = edge.copy()
        swept[0] = sweep
        state, reward, done = hold(env, state, [depth, y, 0.0], swept, 42)
    if not done and abs(retreat) > 1e-6:
        state, reward, done = hold(
            env, state, [depth + retreat, y, 0.0], swept, 24, .025)
    objective, distances = score(state, cube_colors)
    positions = {n: values(state, n, ("x", "y")).round(4).tolist() for n in names}
    bins = {c: values(state, "bin_" + c, ("x", "y")).round(4).tolist()
            for c in COLORS}
    env.close()
    return objective, distances, reward, done, positions, bins, initial_objective, initial_distances


def clipped(genome):
    low = np.array([-.12, -.91, -.95, 1.15, -1.95, .70, -.15, -.08])
    high = np.array([.12, -.76, .95, 1.42, -1.45, 1.30, 1.72, .08])
    return np.clip(genome, low, high)


def main():
    rng = np.random.default_rng(240916)
    # Include the known edge-rake calibration and three broad variants.
    population = [
        # Seed-0 exposed-cube calibration: cube1 is the -y extreme and its
        # positive-q1 contact station is cube_y + 42 mm ~= +16 mm.
        np.array([ .016, -.80, .45, 1.30, -1.70, 1.00, 0.00, 0.00]),
        np.array([ .016, -.80, .70, 1.30, -1.70, 1.00, 0.00, 0.00]),
        np.array([ .016, -.83, .55, 1.25, -1.62, .90, 0.00, .03]),
        # Opposite exposed edge: cube4_y - 42 mm ~= -27 mm.
        np.array([-.027, -.80,-.70, 1.30, -1.70, 1.00, 0.00, 0.00]),
    ]
    results = []
    for generation in range(1):
        generation_results = []
        for index, genome in enumerate(population):
            result = evaluate(genome)
            generation_results.append((result[0], genome.copy(), result))
            print("CAND", generation, index, "obj", round(result[0], 5),
                  "dist", {k: round(v, 4) for k, v in result[1].items()},
                  "reward", result[2], "done", result[3],
                  "genome", np.round(genome, 4).tolist(), flush=True)
        results.extend(generation_results)
        generation_results.sort(key=lambda x: x[0])
        elite = generation_results[0][1]
        scales = np.array([.035, .025, .18, .055, .10, .12, .30, .025])
        population = [elite.copy()] + [clipped(elite + rng.normal(0, scales))
                                       for _ in range(3)]
    results.sort(key=lambda x: x[0])
    obj, genome, result = results[0]
    print("BEST obj", round(obj, 6), "genome", np.round(genome, 6).tolist(),
          "initial_obj", round(result[6], 6),
          "initial_dist", {k: round(v, 6) for k, v in result[7].items()},
          "final_dist", {k: round(v, 6) for k, v in result[1].items()},
          "reward", result[2], "done", result[3],
          "positions", result[4], "bins", result[5])


if __name__ == "__main__":
    main()

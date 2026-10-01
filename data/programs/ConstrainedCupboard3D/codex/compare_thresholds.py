"""Read-only cutoff experiment; dynamically clone policy with one literal changed."""
import concurrent.futures
import sys
import numpy as np
from env_client import make_env


SOURCE = open("approach.py", encoding="utf-8").read()


def run(threshold, seed=1, limit=430):
    ns = {}
    exec(SOURCE.replace("if rx >= 1.75:", f"if rx >= {threshold}:"), ns)
    env = make_env()
    state, info = env.reset(seed=seed)
    policy = ns["GeneratedApproach"](env.action_space, env.observation_space,
                                     env.make_primitives())
    policy.reset(state, info)
    initial = {r.name: np.array([state.get(r, f) for f in ("x", "y", "z")], float)
               for r in policy.rods}
    total = 0.0
    positive = 0.0
    peak_x = {r.name: initial[r.name][0] for r in policy.rods}
    for step in range(limit):
        state, reward, term, trunc, info = env.step(policy.get_action(state))
        total += reward
        positive += max(0.0, reward + .01)
        for r in policy.rods:
            peak_x[r.name] = max(peak_x[r.name], float(state.get(r, "x")))
        if term or trunc:
            break
    final = {r.name: np.array([state.get(r, f) for f in ("x", "y", "z")], float)
             for r in policy.rods}
    env.close()
    return threshold, step + 1, total, positive, initial, final, peak_x, term, trunc


def show(result):
    threshold, steps, total, positive, initial, final, peak_x, term, trunc = result
    summary = {name: {"p0": np.round(initial[name], 3).tolist(),
                      "pf": np.round(final[name], 3).tolist(),
                      "peak_x": round(peak_x[name], 3)} for name in final}
    print("THRESH", threshold, "steps", steps, "return", round(total, 3),
          "positive", round(positive, 3), "done", term, trunc, summary, flush=True)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 430
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs = ((threshold, seed, limit) for threshold in (1.65, 1.70, 1.75, 1.80))
        results = list(pool.map(lambda args: run(*args), jobs))
    for result in results:
        show(result)

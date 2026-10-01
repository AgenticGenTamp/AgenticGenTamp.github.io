"""Find which table object a shorter, non-sweeping arm pose contacts first."""
import sys
import numpy as np

from env_client import make_env
from edge_rake_probe import action_to, get, robot, settle_to


def positions(state):
    return {n: get(state, n, ["x", "y", "z"])
            for n in state.get_object_names()
            if n.startswith("cube") or n.startswith("bin_")}


def run(q4):
    env = make_env()
    state, _ = env.reset(seed=0, options={"object_count": 4})
    start = positions(state)
    q = np.array([0., 1.30, np.pi, q4, 0., 1., 0.])
    state = settle_to(env, state, [.99, .80, np.pi], None, 45, 0)
    state = settle_to(env, state, [-1., .80, np.pi], None, 55, 0)
    state = settle_to(env, state, [-1., .80, 0.], None, 45, 0)
    state = settle_to(env, state, [-1., -.026, 0.], q, 130, 0)
    for step in range(70):
        action = action_to(state, [-.35, -.026, 0.], q, 0)
        action[0] = min(action[0], .012)
        state, reward, term, trunc, _ = env.step(action)
        moved = {n: round(float(np.linalg.norm(p - start[n])), 5)
                 for n, p in positions(state).items()
                 if np.linalg.norm(p - start[n]) > .001}
        if moved:
            print("HIT", q4, step, "base", np.round(robot(state)[:3], 5),
                  "moved", moved, "poses", {n: np.round(positions(state)[n], 4)
                                             for n in moved})
            break
    else:
        print("NO_HIT", q4, "base", np.round(robot(state)[:3], 5))
    env.close()


if __name__ == "__main__":
    run(float(sys.argv[1]) if len(sys.argv) > 1 else -1.)

"""Instrumented seed-0/count-4 rollout of the cardinal edge-rake plan."""
from env_client import make_env
from approach import GeneratedApproach
import numpy as np


COLORS = ("red", "green", "blue", "yellow")


def xy(state, name):
    obj = state.get_object_from_name(name)
    return np.array([float(state.get(obj, "x")), float(state.get(obj, "y"))])


def main():
    env = make_env()
    state, info = env.reset(seed=0, options={"object_count": 4})
    policy = GeneratedApproach(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    targets = {c: xy(state, "bin_" + c).copy() for c in COLORS}
    cube_names = sorted(n for n in state.get_object_names() if n.startswith("cube"))
    offset = min(int(n[4:]) for n in cube_names)
    cube_color = {n: COLORS[(int(n[4:]) - offset) % 4] for n in cube_names}
    reward_changes = []
    previous_reward = None
    terminated = truncated = False
    # The controller has 275-step push legs.  Print each boundary so failures
    # can be attributed to a particular cube/axis rather than only the end.
    for step in range(3400):
        state, reward, terminated, truncated, info = env.step(policy.get_action(state))
        if reward != previous_reward:
            reward_changes.append((step + 1, float(reward)))
            previous_reward = reward
        if (step + 1) % 275 == 0 or terminated or truncated:
            distances = {n: round(float(np.linalg.norm(xy(state, n) - targets[cube_color[n]])), 4)
                         for n in cube_names}
            positions = {n: np.round(xy(state, n), 4).tolist() for n in cube_names}
            print("CHECK", step + 1, "reward", reward, "dist", distances,
                  "pos", positions)
        if terminated or truncated:
            break
    distances = {n: float(np.linalg.norm(xy(state, n) - targets[cube_color[n]]))
                 for n in cube_names}
    print("FINAL", step + 1, "terminated", terminated, "truncated", truncated,
          "reward", float(reward), "reward_changes", reward_changes,
          "distances", {n: round(v, 6) for n, v in distances.items()},
          "positions", {n: np.round(xy(state, n), 6).tolist() for n in cube_names},
          "targets", {c: np.round(v, 6).tolist() for c, v in targets.items()})
    env.close()


if __name__ == "__main__":
    main()

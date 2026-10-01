"""Diagnostic comparison of current adaptive and fixed-positive push sides."""
import argparse
import numpy as np
from approach import GeneratedApproach
from env_client import make_env


class FixedPositive(GeneratedApproach):
    def reset(self, state, info):
        super().reset(state, info)
        self.slot_y = [float("inf")] * len(self.rods)


def xyz(state, obj):
    return np.array([float(state.get(obj, k)) for k in ("x", "y", "z")])


def run(seed, count, fixed):
    env = make_env()
    state, info = env.reset(seed=seed, options={"object_count": count})
    cls = FixedPositive if fixed else GeneratedApproach
    policy = cls(env.action_space, env.observation_space, env.make_primitives())
    policy.reset(state, info)
    init = {o.name: xyz(state, o) for o in policy.rods}
    rows = []
    # Enough to cover all allocated strokes, capped by episode horizon.
    for t in range(min(env.max_steps, 120 + 62 * 12)):
        action = policy.get_action(state)
        state, reward, term, trunc, info = env.step(action)
        if t >= 120 and (t - 119) % 62 == 0:
            rows.append((t + 1, [(o.name, *(xyz(state, o)[:2] - init[o.name][:2]))
                                 for o in policy.rods]))
        if term or trunc:
            break
    final = [(o.name, *xyz(state, o)[:2], *(xyz(state, o)[:2]-init[o.name][:2]))
             for o in policy.rods]
    print("RESULT", "fixed" if fixed else "adapt", seed, count,
          "slots", np.round(policy.slot_y, 3), "init",
          [(o.name, *np.round(init[o.name][:2], 3)) for o in policy.rods])
    print("STROKES", rows)
    print("FINAL", final)
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--count", type=int, required=True); ap.add_argument("--fixed", action="store_true")
    a = ap.parse_args(); run(a.seed, a.count, a.fixed)

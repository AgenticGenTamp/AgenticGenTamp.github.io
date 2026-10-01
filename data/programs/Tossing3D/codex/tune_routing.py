"""Small black-box sweep of routing timings without modifying approach.py."""
import sys
import numpy as np
from env_client import make_env


def xyz(state, name):
    obj = state.get_object_from_name(name)
    return np.array([float(state.get(obj, f)) for f in ("x", "y", "z")])


source = open("approach.py", encoding="utf-8").read()
configs = {
    "baseline": (1.65, 32, 2),
    "early_fast": (1.35, 24, 1),
    "late_fast": (1.90, 24, 1),
    "late_settle": (1.90, 38, 4),
}
chosen = set(sys.argv[1:])
for label, (threshold, setup, coast) in configs.items():
    if chosen and label not in chosen:
        continue
    variant = source.replace("abs(cube[1])<1.65", f"abs(cube[1])<{threshold}")
    variant = variant.replace("else 32", f"else {setup}")
    variant = variant.replace("self.n>=2", f"self.n>={coast}")
    namespace = {"np": np}
    exec(compile(variant, "approach.py", "exec"), namespace)
    Policy = namespace["GeneratedApproach"]
    for seed in (0, 1):
        env = make_env()
        state, info = env.reset(seed=seed, options={"object_count": 1})
        policy = Policy(env.action_space, env.observation_space, {})
        policy.reset(state, info)
        term = False
        max_x = -99.0
        max_abs_y = 0.0
        for step in range(min(1000, env.max_steps)):
            state, reward, term, trunc, info = env.step(policy.get_action(state))
            p = xyz(state, "cube_0")
            max_x = max(max_x, p[0])
            max_abs_y = max(max_abs_y, abs(p[1]))
            if term or trunc:
                break
        p = xyz(state, "cube_0")
        print(label, seed, step + 1, "term", term, "final", np.round(p, 3),
              "max_x", round(max_x, 3), "max_abs_y", round(max_abs_y, 3))
        env.close()

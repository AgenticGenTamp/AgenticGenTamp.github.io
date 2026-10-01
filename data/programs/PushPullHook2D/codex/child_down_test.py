from env_client import make_env
from child_down_variant import GeneratedApproach
import sys
import numpy as np

for seed in map(int, sys.argv[1:]):
    e = make_env(); s, info = e.reset(seed=seed)
    p = GeneratedApproach(e.action_space, e.observation_space, {}); p.reset(s, info)
    best = 99.
    for k in range(e.max_steps):
        s, _, term, trunc, _ = e.step(p.get_action(s))
        d = float(np.linalg.norm(s[20:22]-s[29:31])); best = min(best, d)
        if term or trunc: break
    print(seed, "OK" if term else "FAIL", k+1, round(d,3), round(best,3), p.phase)
    e.close()

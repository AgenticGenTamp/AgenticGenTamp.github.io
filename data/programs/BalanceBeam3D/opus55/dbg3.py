import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for seed in range(int(sys.argv[1]), int(sys.argv[2])):
    env = make_env(); obs, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
    hist = []
    for t in range(1000):
        a = ap.get_action(obs)
        obs, r, te, tr, info = env.step(a)
        hist.append((ap.phase, ap.obj, obs[ap.obj:ap.obj+3].copy() if ap.obj is not None else None, ap.base_t))
        if te: break
    c = obs[38:41]
    print(seed, t, "last phases", [h[0] for h in hist[-3:]], "obj", hist[-1][1], "pos-c", hist[-1][2] - c, "prev pos-c", hist[-2][2] - c)
    env.close()

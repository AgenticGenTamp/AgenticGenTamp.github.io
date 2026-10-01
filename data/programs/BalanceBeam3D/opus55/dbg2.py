import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
np.set_printoptions(precision=3, suppress=True, linewidth=200)
seed, t0, t1 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
env = make_env(); obs, info = env.reset(seed=seed)
print("seesaw", obs[38:41])
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for t in range(t1):
    a = ap.get_action(obs)
    obs, r, te, tr, info = env.step(a)
    if t >= t0:
        p, R = ap.kin.fk(obs[16:19], obs[19:26])
        print(t, ap.phase, "tool", p, "base", obs[16:19], "LB", obs[0:3], "act", a[:3], "grip", obs[26])

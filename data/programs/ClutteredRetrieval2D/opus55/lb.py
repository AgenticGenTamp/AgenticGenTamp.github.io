import numpy as np
from env_client import make_env
from approach import *
env = make_env()
res = {int(l.split()[0]): int(l.split()[3]) for l in open('res_e.txt') if l[0].isdigit()}
tot=0; totlb=0
for seed in range(0, 128):
    obs, info = env.reset(seed=seed)
    if info['object_count'] != 1: continue
    ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
    bc = ap.center('target_block'); rc = ap.center('target_region'); q = ap.q
    d1 = max(0, np.linalg.norm(bc - q[:2]) - 0.37)  # grasp standoff approx (arm .2 + .07 + ...)
    u = (bc - q[:2]) / np.linalg.norm(bc - q[:2]) * d1
    lb = np.abs(u).max() / 0.05 + np.abs(rc - bc).max() / 0.05
    print(seed, res[seed], round(lb, 1))
    tot += res[seed]; totlb += lb
print(tot, totlb)

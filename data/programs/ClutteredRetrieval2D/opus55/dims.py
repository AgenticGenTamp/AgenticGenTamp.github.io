from env_client import make_env
import numpy as np
from approach import GeneratedApproach
env = make_env()
for s in range(5):
    obs, info = env.reset(seed=s)
    ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
    for n in ('target_block','target_region'):
        C = ap.corners(n); print(s, n, round(np.linalg.norm(C[1]-C[0]),4), round(np.linalg.norm(C[3]-C[0]),4))

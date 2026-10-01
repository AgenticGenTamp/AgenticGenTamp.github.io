import numpy as np
from env_client import make_env
np.set_printoptions(precision=5, suppress=True)

env = make_env()
obs, info = env.reset(seed=0)
print("init base", obs[16:19], "arm", obs[19:26], "grip", obs[26])
print("init vels", obs[27:38])
rs = []
prev = obs.copy()
for i in range(30):
    obs, r, term, trunc, info = env.step(np.zeros(11, dtype=np.float32))
    rs.append(r)
    if i < 5 or i == 29:
        print(i, "base", obs[16:19], "arm", obs[19:26], "grip", round(float(obs[26]),5),
              "maxdrift", float(np.abs(obs[16:27]-prev[16:27]).max()), "r", r)
    prev = obs.copy()
print("rewards uniq:", np.unique(np.round(rs,6))[:5], "sum", sum(rs), "term/trunc", term, trunc)
print("info keys", list(info.keys()) if isinstance(info, dict) else info)
print("obj drift large_block", obs[0:3])
env.close()

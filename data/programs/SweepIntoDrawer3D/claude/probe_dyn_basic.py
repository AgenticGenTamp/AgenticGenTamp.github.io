import numpy as np
from env_client import make_env
env = make_env()
obs, info = env.reset(seed=0)
obs = np.asarray(obs, dtype=float)
print("obs shape", obs.shape)
print("robot 125:147", np.round(obs[125:147], 4))
a = np.zeros(11); a[10] = obs[135]
rs = []
for i in range(5):
    o, r, t, tr, inf = env.step(a)
    o = np.asarray(o, float)
    rs.append(r)
print("zero-action rewards", rs)
print("after 5 zero steps", np.round(o[125:147], 4))
print("info keys", list(inf.keys())[:10])
env.close()

import numpy as np
from env_client import make_env

env = make_env()
obs, _ = env.reset(seed=0)
target = obs[19:21].copy()
for axy in ([.4, -.4], [target[0]-.4, -.4], [0., -.03], [0., -.03], [0., -.03]):
    a=np.zeros(11, np.float32); a[:2]=axy
    obs, r, term, trunc, info=env.step(a)
    print(np.round(obs[:2], 6), "distance", float(np.linalg.norm(target-obs[:2])), term, trunc)
    if term or trunc: break
env.close()

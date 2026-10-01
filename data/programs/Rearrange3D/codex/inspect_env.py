import numpy as np
from env_client import make_env

for seed in range(5):
    env = make_env()
    obs, info = env.reset(seed=seed)
    print(seed, "objects", np.round(obs[[0,1,2,13,14,15,16,17,18,29,30,31,32,33,34,45,46,47]],3),
          "robot", np.round(obs[93:104],3), "info", info)
    env.close()

env = make_env(); obs, _ = env.reset(seed=0)
for idx in range(11):
    a = np.zeros(11, np.float32); a[idx] = 0.1 if idx < 10 else 0.0
    before = obs.copy(); obs,r,t,tr,info=env.step(a)
    changed=np.where(np.abs(obs-before)>1e-4)[0]
    print("action",idx,"r",r,"changed",changed.tolist(),"delta",np.round((obs-before)[changed],4).tolist(),"rob",np.round(obs[93:115],3))
env.close()

import numpy as np
from env_client import make_env
env = make_env()
for a in [[0,0.05,0,0,0],[0,0.0499,0,0,0],[0,-0.05,0,0,0],[0.05,0,0,0,0],[0,0,0.0654,0,0],[0,0,0,0.1,0],[0,0,0,0.0999,0],[0,0,0,0,0.02],[0,0,0,0,0.0199]]:
    obs, info = env.reset(seed=1)
    try:
        obs,r,t,tr,info=env.step(np.array(a,dtype=float)); print(a,'ok',r,t)
    except Exception as e: print(a,'ERR',str(e)[:100])

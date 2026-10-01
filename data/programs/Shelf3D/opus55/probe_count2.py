from env_client import make_env
from helpers import *
env=make_env()
for k in [6,8]:
  for seed in [0,1]:
    obs,info=env.reset(seed=seed,options={'object_count':k})
    c=cubes(obs); s=rstate(obs)
    print(k,seed,len(c),s[:3].round(2),[v[:2].round(2).tolist() for v in c.values()])

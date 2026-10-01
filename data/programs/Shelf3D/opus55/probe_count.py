from env_client import make_env
from helpers import *
env=make_env()
for seed in range(0,40,3):
    obs,info=env.reset(seed=seed)
    c=cubes(obs); s=rstate(obs)
    print(seed,len(c),s[:3].round(2),[v[:2].round(2).tolist() for v in c.values()])

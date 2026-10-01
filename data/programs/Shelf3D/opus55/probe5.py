from env_client import make_env
from helpers import *
env=make_env()
for seed in [0,2]:
    obs,info=env.reset(seed=seed)
    n=sorted(cubes(obs))[0]
    obs=pick(env,obs,n)
    print(seed,n,cubes(obs)[n][:3].round(3), rstate(obs)[:3].round(3))

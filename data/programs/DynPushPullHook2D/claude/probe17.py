from env_client import make_env
from ctl import act, rget, oget
from grasp import grasp_seq, wrap
import numpy as np
env=make_env()
for seed in range(8):
    obs,info=env.reset(seed=seed)
    obs,n,ok=grasp_seq(env,obs)
    print(seed,info,"grasp",ok,"steps",n, "hookmove")
env.close()

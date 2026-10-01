import numpy as np
from env_client import make_env
from ctl import act, rget, oget
from grasp import grasp_seq
env=make_env()
for wait in [0, 100, 200, 400]:
    obs,info=env.reset(seed=42)
    for _ in range(wait): obs,_,_,_,_=env.step(act())
    obs2,n,ok=grasp_seq(env,obs)
    print("wait",wait,"grasp",ok,"steps",n)
env.close()

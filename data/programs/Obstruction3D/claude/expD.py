import numpy as np
from env_client import make_env
import robot
from s2 import *
env=make_env(); obs,info=env.reset(seed=1)
Z=0.10
def probe(x,y):
    global obs
    obs2,bl,err=go_world(env,obs,np.array([x,y,0.30]),robot.down_R(0.0),nmax=60)
    obs=obs2
    obs2,bl,err=go_world(env,obs,np.array([x,y,Z]),robot.down_R(0.0),nmax=25)
    obs=obs2
    r = 'TABLE' if bl else 'free'
    obs2,bl2,err2=go_world(env,obs,np.array([x,y,0.30]),robot.down_R(0.0),nmax=40)
    obs=obs2
    return r
for y in [0.30,0.36,0.40,0.44]:
    print("x=0.25 y",y,probe(0.25,y),flush=True)
for x in [0.02,0.05,0.44,0.48,0.52]:
    print("y=0.0 x",x,probe(x,0.0),flush=True)

import numpy as np
from env_client import make_env
from lib_util import robot, step_to
env=make_env()
obs,_=env.reset(seed=1); q=robot(obs)[3:10]
for tgt in [[-9,0,0],[9,0,0],[0,9,0],[0,-9,0]]:
    obs,_=env.reset(seed=1)
    obs,rej,n=step_to(env,obs,tgt,q,maxsteps=120)
    print(tgt,"->",np.round(robot(obs)[:3],3),"rej",rej,"steps",n)
env.close()

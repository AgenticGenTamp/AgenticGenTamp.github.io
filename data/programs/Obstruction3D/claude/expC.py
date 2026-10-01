import numpy as np
from env_client import make_env
import robot
from s2 import *
env=make_env(); obs,info=env.reset(seed=0)
# min gripper z over free table
for xy in [(0.30,-0.25),(0.20,0.0)]:
    zmin=None
    obs,bl,err=go_world(env,obs,np.array([xy[0],xy[1],0.30]),robot.down_R(0.0),nmax=80)
    for Z in np.arange(0.28,0.05,-0.005):
        obs,bl,err=go_world(env,obs,np.array([xy[0],xy[1],Z]),robot.down_R(0.0),nmax=15)
        if bl: zmin=Z+0.005; break
    print("xy",xy,"min gripper z",round(zmin,3) if zmin else None, flush=True)
    obs,bl,err=go_world(env,obs,np.array([xy[0],xy[1],0.30]),robot.down_R(0.0),nmax=40)
# table extent probe at z=0.15 (below any object top? no, just table)
for y in [0.30,0.35,0.40,0.45]:
    obs,bl,err=go_world(env,obs,np.array([0.25,y,0.30]),robot.down_R(0.0),nmax=60)
    obs,bl,err=go_world(env,obs,np.array([0.25,y,0.13]),robot.down_R(0.0),nmax=30)
    print("y",y,"blocked",bl,"err",round(err,3),flush=True)
for x in [0.42,0.46,0.50]:
    obs,bl,err=go_world(env,obs,np.array([x,0.0,0.30]),robot.down_R(0.0),nmax=60)
    obs,bl,err=go_world(env,obs,np.array([x,0.0,0.13]),robot.down_R(0.0),nmax=30)
    print("x",x,"blocked",bl,"err",round(err,3),flush=True)

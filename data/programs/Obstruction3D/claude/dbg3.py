import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
import robot
from s2 import *
env=make_env(); obs,info=env.reset(seed=39)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for i in range(120):
    a=ap.get_action(obs); obs,r,t,tr,info=env.step(a)
q=robot_q(obs)
print("stage",ap.stage,"q",np.round(q,3),"tool",np.round(ap._tool(obs)[:3,3],4))
for j in range(7):
    for sgn in [1,-1]:
        a=np.zeros(11,dtype=np.float32); a[3+j]=0.05*sgn
        o2,r,t,tr,inf=env.step(a)
        ok=np.max(np.abs(robot_q(o2)-(q+a[3:10])))<1e-6
        print("j",j+1,sgn,ok,end="; ")
print()

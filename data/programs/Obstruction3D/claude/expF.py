import numpy as np
from env_client import make_env
from approach import GeneratedApproach
import robot
from s2 import *
env=make_env(); obs,info=env.reset(seed=2)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for i in range(19):
    a=ap.get_action(obs); obs,r,t,tr,info=env.step(a)
print("stage",ap.stage,"hold",rinfo(obs)['grasp_active'],"tool",np.round(ap._tool(obs)[:3,3],4))
q0=robot_q(obs)
rng=np.random.default_rng(0)
moved=0
for k in range(40):
    a=np.zeros(11,dtype=np.float32); a[3:10]=rng.uniform(-0.2,0.2,7)
    obs2,r,t,tr,inf=env.step(a)
    if np.max(np.abs(robot_q(obs2)-q0))>1e-6:
        print("random move accepted at",k, np.round(a[3:10],3)); moved+=1; obs=obs2; q0=robot_q(obs); break
print("random moves accepted:",moved)
# try single-joint moves
for j in range(7):
    for sgn in [1,-1]:
        a=np.zeros(11,dtype=np.float32); a[3+j]=0.05*sgn
        obs2,r,t,tr,inf=env.step(a)
        if np.max(np.abs(robot_q(obs2)-q0))>1e-6:
            print("joint",j+1,"sgn",sgn,"ok")
# try base
for idx in [0,1,2]:
    for sgn in [1,-1]:
        a=np.zeros(11,dtype=np.float32); a[idx]=0.05*sgn
        obs2,r,t,tr,inf=env.step(a)
        if np.max(np.abs(base(obs2)-base(obs)))>1e-6: print("base",idx,sgn,"ok")
# try release
a=np.zeros(11,dtype=np.float32); a[10]=1.0
obs2,r,t,tr,inf=env.step(a)
print("release ->",rinfo(obs2)['grasp_active'])

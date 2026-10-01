import numpy as np
from env_client import make_env
from approach import GeneratedApproach
import robot
from s2 import *
def mk(nsteps=19):
    env=make_env(); obs,info=env.reset(seed=2)
    ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
    for i in range(nsteps):
        a=ap.get_action(obs); obs,r,t,tr,info=env.step(a)
    return env,obs,ap
env,obs,ap=mk()
q0=robot_q(obs)
print("q",np.round(q0,3),"tool",np.round(ap._tool(obs)[:3,3],4))
for j in range(7):
    for sgn in [1,-1]:
        for mag in [0.02,0.1]:
            a=np.zeros(11,dtype=np.float32); a[3+j]=mag*sgn
            obs2,r,t,tr,inf=env.step(a)
            ok=np.max(np.abs(robot_q(obs2)-(q0+a[3:10])))<1e-6
            print("joint",j+1,"sgn",sgn,"mag",mag,"->",ok, end="  ")
    print()
# what if we open the gripper first (before it was grasped)? test grasp at 18 steps (before close)
env2,obs2,ap2=mk(17)
print("pre-close stage",ap2.stage,"tool",np.round(ap2._tool(obs2)[:3,3],4),"hold",rinfo(obs2)['grasp_active'])
q0=robot_q(obs2)
for j in range(7):
    a=np.zeros(11,dtype=np.float32); a[3+j]=0.1
    o3,r,t,tr,inf=env2.step(a)
    print("  pre-close joint",j+1,"->",np.max(np.abs(robot_q(o3)-(q0+a[3:10])))<1e-6, end="")
print()

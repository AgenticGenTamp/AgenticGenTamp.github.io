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
print("tool",np.round(ap._tool(obs)[:3,3],4))
for j in range(7):
    row=[]
    for sgn in [1,-1]:
        a=np.zeros(11,dtype=np.float32); a[3+j]=0.05*sgn
        o2,r,t,tr,inf=env.step(a)
        q2=robot_q(o2)
        ok=np.max(np.abs(q2-(q+a[3:10])))<1e-6
        row.append((sgn,ok))
        if ok:  # undo
            a2=np.zeros(11,dtype=np.float32); a2[3+j]=-0.05*sgn
            o3,r,t,tr,inf=env.step(a2)
            q=robot_q(o3)
    print("joint",j+1,row)
# cartesian dirs
T=ap._tool(obs)
for nm,d in [("+x",[0.03,0,0]),("-x",[-0.03,0,0]),("+y",[0,0.03,0]),("-y",[0,-0.03,0]),("+z",[0,0,0.03]),("-z",[0,0,-0.03])]:
    tgt=T[:3,3]+np.array(d)
    B=robot.base_tf(*base(obs)); pl=(np.linalg.inv(B)@np.append(tgt,1.0))[:3]; Rl=B[:3,:3].T@robot.down_R(0.0)
    dq,ep,ew=robot.ik_step(q,pl,Rl,damp=0.03,max_delta=0.05)
    a=np.zeros(11,dtype=np.float32); a[3:10]=dq
    o2,r,t,tr,inf=env.step(a); q2=robot_q(o2)
    ok=np.max(np.abs(q2-(q+dq)))<1e-6
    print(nm,ok)
    if ok:
        a2=np.zeros(11,dtype=np.float32); a2[3:10]=-dq
        o3,r,t,tr,inf=env.step(a2); q=robot_q(o3)

import numpy as np
from env_client import make_env
from approach import GeneratedApproach
import robot
from s2 import *
def mk(nsteps):
    env=make_env(); obs,info=env.reset(seed=4, options={"object_count":6})
    ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
    for i in range(nsteps):
        a=ap.get_action(obs); obs,r,t,tr,info=env.step(a)
    return env,obs,ap
env,obs,ap=mk(30)
q=robot_q(obs); T=ap._tool(obs)
print("tool",np.round(T[:3,3],4),"hold",rinfo(obs)['grasp_active'],"yaw",round(np.arctan2(T[1,0],T[0,0]),3))
for nm,d in [("+z",[0,0,0.02]),("-z",[0,0,-0.02]),("+x",[0.02,0,0]),("-x",[-0.02,0,0]),("+y",[0,0.02,0]),("-y",[0,-0.02,0])]:
    for md in [0.06,0.01]:
        tgt=T[:3,3]+np.array(d)
        B=robot.base_tf(*base(obs)); pl=(np.linalg.inv(B)@np.append(tgt,1.0))[:3]; Rl=B[:3,:3].T@T[:3,:3]
        dq,ep,ew=robot.ik_step(q,pl,Rl,damp=0.03,max_delta=md)
        a=np.zeros(11,dtype=np.float32); a[3:10]=dq
        o2,r,t,tr,inf=env.step(a); q2=robot_q(o2)
        ok=np.max(np.abs(q2-(q+dq)))<1e-6
        print(nm,md,ok,end="; ")
        if ok:
            a2=np.zeros(11,dtype=np.float32); a2[3:10]=-dq
            o3,r,t,tr,inf=env.step(a2); q=robot_q(o3)
    print()
# random wiggle test
cnt=0
rng=np.random.default_rng(0)
for k in range(30):
    dq=rng.uniform(-0.05,0.05,7)
    a=np.zeros(11,dtype=np.float32); a[3:10]=dq
    o2,r,t,tr,inf=env.step(a)
    if np.max(np.abs(robot_q(o2)-(q+dq)))<1e-6:
        cnt+=1
        a2=np.zeros(11,dtype=np.float32); a2[3:10]=-dq
        o3,r,t,tr,inf=env.step(a2); q=robot_q(o3)
print("random accepted",cnt,"/30")

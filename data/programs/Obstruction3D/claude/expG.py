import numpy as np
from env_client import make_env
from approach import GeneratedApproach
import robot
from s2 import *
def mk():
    env=make_env(); obs,info=env.reset(seed=2)
    ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
    for i in range(19):
        a=ap.get_action(obs); obs,r,t,tr,info=env.step(a)
    return env,obs,ap
for md in [0.2,0.1,0.05,0.02,0.01]:
    env,obs,ap=mk()
    q=robot_q(obs); ok=0
    T=ap._tool(obs); tgt=np.array([T[0,3],T[1,3],0.28])
    for k in range(60):
        b=base(obs); B=robot.base_tf(*b)
        pl=(np.linalg.inv(B)@np.append(tgt,1.0))[:3]; Rl=B[:3,:3].T@robot.down_R(0.0)
        dq,ep,ew=robot.ik_step(q,pl,Rl,damp=0.03,max_delta=md)
        a=np.zeros(11,dtype=np.float32); a[3:10]=dq
        obs,r,t,tr,inf=env.step(a); qn=robot_q(obs)
        if np.max(np.abs(qn-(q+dq)))>1e-6: break
        q=qn; ok+=1
        if ep<0.005: break
    print("max_delta",md,"steps ok",ok,"final tool z",round(ap._tool(obs)[2,3],4),flush=True)
    env.close()
# from deadlock: try moving DOWN then up, or lateral
env,obs,ap=mk()
q=robot_q(obs)
for name,delta in [("down",np.array([0,0,-0.01])),("+x",np.array([0.02,0,0])),("-x",np.array([-0.02,0,0])),("+y",np.array([0,0.02,0])),("-y",np.array([0,-0.02,0]))]:
    T=ap._tool(obs); tgt=T[:3,3]+delta
    b=base(obs); B=robot.base_tf(*b)
    pl=(np.linalg.inv(B)@np.append(tgt,1.0))[:3]; Rl=B[:3,:3].T@robot.down_R(0.0)
    dq,ep,ew=robot.ik_step(q,pl,Rl,damp=0.03,max_delta=0.05)
    a=np.zeros(11,dtype=np.float32); a[3:10]=dq
    obs2,r,t,tr,inf=env.step(a)
    okk=np.max(np.abs(robot_q(obs2)-(q+dq)))<1e-6
    print(name,"accepted",okk)

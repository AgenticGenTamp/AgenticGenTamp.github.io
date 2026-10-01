import numpy as np
from env_client import make_env
from approach import GeneratedApproach
import robot
from s2 import *
env=make_env(); obs,info=env.reset(seed=39)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for i in range(150):
    a=ap.get_action(obs); obs,r,t,tr,info=env.step(a)
q=robot_q(obs); T=ap._tool(obs)
print("stage",ap.stage,"task",ap.tasks[0]['obj'],"yaw_try",ap.yaw_try,"cur_yaw",getattr(ap,'cur_yaw',None))
print("q",np.round(q,3),"tool",np.round(T[:3,3],4))
print("tool R",np.round(T[:3,:3],3))
a=ap.get_action(obs)
print("action",np.round(a,4))
p=ap._pos(obs,'target_block'); tgt=np.array([p[0],p[1],0.28])
B=robot.base_tf(*base(obs)); pl=(np.linalg.inv(B)@np.append(tgt,1.0))[:3]
for yaw in [0.0, getattr(ap,'cur_yaw',0.0)]:
    Rl=B[:3,:3].T@robot.down_R(yaw)
    dq,ep,ew=robot.ik_step(q,pl,Rl,damp=0.03,max_delta=0.2)
    print("yaw",round(yaw,3),"ep",round(ep,4),"ew",round(ew,4),"dq",np.round(dq,4))

print("--- applying approach actions and watching q ---")
for k in range(12):
    a=ap.get_action(obs)
    q0=robot_q(obs)
    obs,r,t,tr,info=env.step(a)
    q1=robot_q(obs)
    print(k,"esc",ap.escape,"stage",ap.stage,"|a|",np.round(np.abs(a[3:10]).max(),3),"moved",np.round(np.abs(q1-q0).max(),4),"tool",np.round(ap._tool(obs)[:3,3],4))

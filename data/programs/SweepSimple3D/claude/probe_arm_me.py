# I control arm to a config and watch which joints move; also check gripper
from env_client import make_env
import numpy as np
env=make_env()
obs,info=env.reset(seed=6)
def g(n,f): return float(obs.get(obs.get_object_from_name(n),f))
def joints(): return np.array([g("robot","pos_arm_joint%d"%i) for i in range(1,8)])
print("start",np.round(joints(),3), "grip",round(g("robot","pos_gripper"),3))
tgt=np.array([0.0,0.4,3.142,-2.0,0.0,-0.6,1.571])
for i in range(60):
    a=np.zeros(11,dtype=np.float32)
    a[3:10]=np.clip(tgt-joints(),-0.1,0.1)
    a[10]=1.0
    obs,rew,term,trunc,info=env.step(a)
    if i%15==0: print(i,np.round(joints(),3),round(g("robot","pos_gripper"),3), round(rew,3))
print("final",np.round(joints(),3),"grip",round(g("robot","pos_gripper"),3))
env.close()

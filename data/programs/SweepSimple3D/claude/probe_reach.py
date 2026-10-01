# Sweep joint2 (shoulder) through range with joint4 varied, check if arm hits floor/limits.
from env_client import make_env
import numpy as np
env=make_env()
obs,info=env.reset(seed=6)
def g(n,f): return float(obs.get(obs.get_object_from_name(n),f))
def J(): return np.array([g("robot","pos_arm_joint%d"%i) for i in range(1,8)])
def goto(tgt,n=45):
    global obs
    for _ in range(n):
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(tgt-J(),-0.1,0.1)
        obs,rew,t,tr,i2=env.step(a)
    return J()
cfgs=[
 [0,1.2,3.142,-1.2,0,-1.2,1.571],
 [0,1.5,3.142,-0.8,0,-1.5,1.571],
 [0,0.6,3.142,-2.2,0,-0.4,1.571],
]
for c in cfgs:
    res=goto(np.array(c))
    print("target",np.round(c,2),"-> actual",np.round(res,3),"err",np.round(res-np.array(c),3))
env.close()

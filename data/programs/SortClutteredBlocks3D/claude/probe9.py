from env_client import make_env
import numpy as np
for mag in [0.02,0.05,0.1]:
    env=make_env(); obs,info=env.reset(seed=0)
    o=obs.get_object_from_name('robot'); s=float(obs.get(o,'pos_arm_joint1')); sb=float(obs.get(o,'pos_base_x'))
    a=np.zeros(11,dtype=np.float32); a[3]=mag; a[0]=mag
    vals=[];bs=[]
    for t in range(12):
        obs,r,te,tr,i=env.step(a); o=obs.get_object_from_name('robot')
        vals.append(float(obs.get(o,'pos_arm_joint1'))-s); bs.append(float(obs.get(o,'pos_base_x'))-sb)
    print('mag',mag,'j1',[round(v,4) for v in vals])
    print('        base',[round(v,4) for v in bs])
    env.close()

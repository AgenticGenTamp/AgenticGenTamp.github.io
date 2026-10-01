from env_client import make_env
import numpy as np
env=make_env(); obs,info=env.reset(seed=0)
a=np.zeros(11,dtype=np.float32)
vals=[]
for t in range(15):
    a[10]=1.0; obs,r,te,tr,i=env.step(a); o=obs.get_object_from_name('robot'); vals.append(round(float(obs.get(o,'pos_gripper')),3))
print('close',vals)
vals=[]
for t in range(15):
    a[10]=0.0; obs,r,te,tr,i=env.step(a); o=obs.get_object_from_name('robot'); vals.append(round(float(obs.get(o,'pos_gripper')),3))
print('open',vals)
env.close()

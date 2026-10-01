from env_client import make_env
import numpy as np
env = make_env()
def rs(obs):
    R=obs.get_object_from_name('robot')
    return np.array([float(obs.get(R,k)) for k in ['pos_base_x','pos_base_y','pos_base_rot']+[f'joint_{i}' for i in range(1,8)]+['finger_state','grasp_active']])
def stepa(**kw):
    a=np.zeros(11,np.float32)
    for k,v in kw.items(): a[int(k[1:])]=v
    return env.step(a)
# truncation
obs,_=env.reset(seed=0); n=0
while n<1500:
    obs,r,te,tr,info=stepa(a3=0.0); n+=1
    if te or tr: break
print("ended at step",n,"term",te,"trunc",tr, "r",r)

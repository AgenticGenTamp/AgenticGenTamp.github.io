from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
R = obs.get_object_from_name('robot')
RF = ['pos_base_x','pos_base_y','pos_base_rot']+[f'joint_{i}' for i in range(1,8)]+['finger_state','grasp_active']
def rs(o): return np.round([o.get(R,f) for f in RF],4)
print(rs(obs))
for i in range(11):
    obs,_=env.reset(seed=0)
    a=np.zeros(11,dtype=np.float32); a[i]=0.2 if i<10 else -1
    o2,r,t,tr,inf=env.step(a)
    print(i, rs(o2)-rs(obs), r,t)
# large action
obs,_=env.reset(seed=0)
a=np.zeros(11,dtype=np.float32); a[3]=1.0
o2,*_=env.step(a); print('big', rs(o2)-rs(obs))
env.close()

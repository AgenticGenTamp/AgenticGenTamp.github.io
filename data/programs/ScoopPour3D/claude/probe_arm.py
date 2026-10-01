from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=1)
r = obs.get_object_from_name('robot')
def q(o): return np.round(o.data[r][3:11],3)
print('init', q(obs))
for j in range(7):
    a=np.zeros(11,dtype=np.float32); a[3+j]=0.1
    for i in range(20): obs,_,_,_,_=env.step(a)
    print('joint',j,'+20x0.1 ->', q(obs))
    a=np.zeros(11,dtype=np.float32); a[3+j]=-0.1
    for i in range(20): obs,_,_,_,_=env.step(a)
    print('joint',j,'back ->', q(obs))
# gripper
for g in [1.0, 0.0, 0.5]:
    a=np.zeros(11,dtype=np.float32); a[10]=g
    for i in range(10): obs,_,_,_,_=env.step(a)
    print('grip',g,'->',q(obs)[7])
env.close()

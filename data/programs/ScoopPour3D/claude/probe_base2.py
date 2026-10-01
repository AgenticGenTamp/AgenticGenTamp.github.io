from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=5, options={'object_count':10})
R = obs.get_object_from_name('robot')
b = lambda: np.round(obs.data[R][:3],3)
a=np.zeros(11,dtype=np.float32); a[2]=0.1
for i in range(12): obs,_,_,_,_=env.step(a)
print('after rot', b())
a=np.zeros(11,dtype=np.float32); a[0]=0.1
p0=obs.data[R][:3].copy()
for i in range(5): obs,_,_,_,_=env.step(a)
print('cmd x+ moved', np.round(obs.data[R][:3]-p0,3))
a=np.zeros(11,dtype=np.float32); a[1]=0.1
p0=obs.data[R][:3].copy()
for i in range(5): obs,_,_,_,_=env.step(a)
print('cmd y+ moved', np.round(obs.data[R][:3]-p0,3))
# how far can base go toward counter? drive x+ a lot
a=np.zeros(11,dtype=np.float32); a[2]=-0.1
for i in range(12): obs,_,_,_,_=env.step(a)
print('unrot', b())
a=np.zeros(11,dtype=np.float32); a[0]=0.1
for i in range(20): obs,_,_,_,_=env.step(a)
print('drive +x 20 steps ->', b())
env.close()

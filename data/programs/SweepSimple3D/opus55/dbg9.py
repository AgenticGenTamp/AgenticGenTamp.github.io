import numpy as np
from env_client import make_env
env=make_env(); obs,_=env.reset(seed=3, options={'object_count':10})
R=obs.get_object_from_name('robot')
def base(): return np.array([obs.get(R,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
def go(bt,n):
    global obs
    for _ in range(n):
        d=np.array(bt)-base(); a=np.zeros(11,np.float32); a[:2]=np.clip(d[:2]/0.87,-.1,.1); a[2]=np.clip(d[2],-.1,.1)
        obs,*_=env.step(a)
    return base().round(3)
print(go([1.364,1.6,-np.pi/2],20))
print(go([1.364,1.349,-np.pi/2],20))
print(go([1.364,1.0,-np.pi/2],20))
print(go([1.30,0.8,-np.pi/2],20))
print(go([1.2,0.8,-np.pi/2],20))

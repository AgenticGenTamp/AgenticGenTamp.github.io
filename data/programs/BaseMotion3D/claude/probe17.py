import numpy as np
from env_client import make_env
env=make_env()
def miny(x, seed=0):
    o,_=env.reset(seed=seed)
    for k in range(30):
        a=np.zeros(11); a[0]=np.clip(x-o[0],-0.4,0.4); a[1]=np.clip(-1.7-o[1],-0.4,0.4)
        if np.all(np.abs(a)<1e-7): break
        o,r,t,tr,_=env.step(a)
    lo,hi=-3.0,float(o[1])  # lo unreachable guess
    for _ in range(22):
        mid=(lo+hi)/2
        a=np.zeros(11); a[1]=np.clip(mid-o[1],-0.4,0.4)
        o2,r,t,tr,_=env.step(a)
        if abs(float(o2[1])-mid)<1e-6: hi=mid; o=o2
        else: lo=mid
    return hi
for x in [-1.2,-1.0,-0.9,-0.85,-0.8,-0.78,-0.76,-0.74,-0.7,-0.5,0.0,0.5,1.0,1.3,1.4,1.42,1.44,1.46,1.5,1.6,2.0]:
    print(round(x,2), round(miny(x),4))

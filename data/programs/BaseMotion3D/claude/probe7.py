import numpy as np, time
from env_client import make_env
env=make_env()
def term_at(seed, gx, gy, joints=None, rot=0.0):
    o,_=env.reset(seed=seed)
    for k in range(30):
        a=np.zeros(11)
        a[0]=np.clip(gx-o[0],-0.4,0.4); a[1]=np.clip(gy-o[1],-0.4,0.4)
        a[2]=np.clip(rot-o[2],-0.4,0.4)
        if joints is not None:
            for j in range(7):
                a[3+j]=np.clip(joints[j]-o[3+j],-0.4,0.4)
        o,r,t,tr,_=env.step(a)
        if t: return True
        if np.all(np.abs(a[:10])<1e-7): return False
    return False
t0=time.time()
o,_=env.reset(seed=0); tx,ty=float(o[19]),float(o[20])
for ang in [0,45,90,135,180,225,270,315]:
    th=np.radians(ang)
    lo,hi=0.0,1.0
    for _ in range(18):
        mid=(lo+hi)/2
        if term_at(0, tx-mid*np.cos(th), ty-mid*np.sin(th)): lo=mid
        else: hi=mid
    print(ang, round(lo,4))
print("time",time.time()-t0)

import numpy as np
from env_client import make_env

env=make_env()
rows=[]
for h in [0.,.4,.6,.8,1.]:
    s,_=env.reset(seed=779)
    for k in range(5):
        a=np.zeros(11,dtype=np.float32)
        a[2]=np.clip(h-s[2],-.4,.4)
        s,*_=env.step(a)
        if abs(h-s[2])<1e-6:break
    dest=np.array([-.7,-1.6])
    for k in range(5):
        a=np.zeros(11,dtype=np.float32);a[:2]=np.clip(dest-s[:2],-.4,.4)
        s,*_=env.step(a)
        if np.linalg.norm(s[:2]-dest)<1e-6:break
    lo,hi=-2.1,float(s[1])
    for k in range(20):
        y=(lo+hi)/2
        a=np.zeros(11,dtype=np.float32);a[1]=y-s[1]
        nxt,*_=env.step(a)
        if abs(nxt[1]-s[1])<1e-7:lo=y
        else:hi=float(nxt[1])
        s=nxt
    rows.append((h,float(s[0]),hi))
    print(rows[-1],flush=True)
A=np.array([[-np.sin(h),np.cos(h),1] for h,x,y in rows])
b=np.array([y*np.cos(h)-x*np.sin(h) for h,x,y in rows])
fit=np.linalg.lstsq(A,b,rcond=None)[0]
print('FIT',fit.tolist(),'RESIDUAL',(A@fit-b).tolist(),flush=True)
env.close()

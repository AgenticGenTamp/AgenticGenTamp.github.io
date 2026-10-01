import numpy as np
from env_client import make_env

env=make_env()
for x in np.arange(-.86,-.589,.005):
    s,_=env.reset(seed=779)
    for waypoint in [[x,0],[x,-1.85]]:
        for i in range(10):
            a=np.zeros(11,dtype=np.float32)
            a[:2]=np.clip(np.array(waypoint)-s[:2],-.4,.4)
            s,r,done,trunc,_=env.step(a)
            if np.max(np.abs(s[:2]-waypoint))<1e-5:break
    lo,hi=-2.1,float(s[1])
    for i in range(13):
        y=(lo+hi)/2
        a=np.zeros(11,dtype=np.float32)
        a[1]=y-s[1]
        nxt,r,done,trunc,_=env.step(a)
        if abs(nxt[1]-s[1])<1e-7:lo=y
        else:hi=float(nxt[1])
        s=nxt
    print(round(float(x),5),round(hi,6),flush=True)
env.close()

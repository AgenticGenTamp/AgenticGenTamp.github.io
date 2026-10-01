import numpy as np
from env_client import make_env
for x in (-2,-1.5,-1,-.9,-.8,-.7,-.65,-.6,-.5,0,.5,1,1.3,1.4,1.5,1.6,1.7,2):
    env=make_env();s,_=env.reset(seed=0)
    for k in range(8):
        a=np.zeros(11,dtype=np.float32);a[:2]=np.clip(np.array([x,-1.85])-s[:2],-.4,.4)
        s,r,t,tr,i=env.step(a)
        if np.linalg.norm(s[:2]-[x,-1.85])<1e-6:break
    blocked=False
    for k in range(25):
        a=np.zeros(11,dtype=np.float32);a[1]=-.01
        n,r,t,tr,i=env.step(a)
        if np.max(np.abs(n-s))<1e-6:blocked=True;break
        s=n
    print('X',x,'lastsafe',s[:2].tolist(),'blocked',blocked,flush=True)
    env.close()

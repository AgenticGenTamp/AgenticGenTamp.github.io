import numpy as np
from env_client import make_env
for y in (-1.91,-1.94,-1.98,-2.):
    for side,start,dx in (('left',-.9,.01),('right',1.6,-.01)):
        env=make_env();s,_=env.reset(seed=0)
        for k in range(8):
            a=np.zeros(11,dtype=np.float32);a[:2]=np.clip(np.array([start,y])-s[:2],-.4,.4)
            s,r,t,tr,i=env.step(a)
            if np.linalg.norm(s[:2]-[start,y])<1e-6:break
        for k in range(40):
            a=np.zeros(11,dtype=np.float32);a[0]=dx
            n,r,t,tr,i=env.step(a)
            if np.max(np.abs(n-s))<1e-6:break
            s=n
        print('CORNER',side,y,'safe',s[:2].tolist(),'nextblocked',k<39,flush=True)
        env.close()

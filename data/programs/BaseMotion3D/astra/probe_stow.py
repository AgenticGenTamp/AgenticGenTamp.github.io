import numpy as np
from env_client import make_env
cases=[('rot',v) for v in (-3.14,-2.4,-1.57,-.8,.8,1.57,2.4,3.14)]+[(str(j),v) for j in range(3,10) for v in (-1.2,1.2)]
for name,value in cases:
    env=make_env();s,_=env.reset(seed=779)
    for k in range(8):
        a=np.zeros(11,dtype=np.float32);j=2 if name=='rot' else int(name)
        if k==0:orig=float(s[j])
        a[j]=np.clip(orig+value-s[j],-.4,.4)
        n,r,t,tr,i=env.step(a)
        if np.max(np.abs(n-s))<1e-6:break
        s=n
    config=s[:10].copy()
    for k in range(8):
        a=np.zeros(11,dtype=np.float32);a[:2]=np.clip(s[19:21]-s[:2],-.4,.4)
        n,r,t,tr,i=env.step(a)
        if t or tr or np.max(np.abs(n-s))<1e-6:s=n;break
        s=n
    print('CASE',name,value,'config',config.tolist(),'end',s[:3].tolist(),'term',t,flush=True)
    env.close()

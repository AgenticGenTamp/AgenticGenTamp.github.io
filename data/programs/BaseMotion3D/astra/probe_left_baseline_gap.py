import numpy as np
from env_client import make_env
for seed in (20980,29069,32894,34027):
    env=make_env();s,_=env.reset(seed=seed)
    aim=s[19:21]+[-.0353546,.0353546]
    for k in range(8):
        a=np.zeros(11,dtype=np.float32);a[:2]=np.clip(aim-s[:2],-.4,.4);a[2]=np.clip(np.pi/4-s[2],-.4,.4)
        n,r,t,tr,i=env.step(a)
        changed=np.max(np.abs(n[:3]-s[:3]))>1e-7;s=n
        if t or not changed:break
    print(seed,'baseline',t,'steps',k+1,'pos',s[:3].tolist(),flush=True)
    env.close()

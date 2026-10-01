import numpy as np
from env_client import make_env
selected=[]
for seed in range(1000000):
    p=np.random.default_rng(seed).uniform(-2,2,2)
    if -.65<p[0]<-.5 and p[1]<-1.959:
        selected.append((seed,p))
        if len(selected)==15:break
for seed,p in selected:
    env=make_env();s,_=env.reset(seed=seed)
    d=s[19:21]-np.array([-.5,-2.1]);dist=np.linalg.norm(d);normal=d/dist
    heading=np.arctan2(normal[1],normal[0])-np.pi/2
    aim=s[19:21]+.049999*normal
    for k in range(10):
        a=np.zeros(11,dtype=np.float32);a[:2]=np.clip(aim-s[:2],-.4,.4);a[2]=np.clip(heading-s[2],-.4,.4)
        n,r,t,tr,i=env.step(a)
        changed=np.max(np.abs(n[:3]-s[:3]))>1e-7;s=n
        if t or not changed:break
    print(seed,p.tolist(),'d',dist,'heading',heading,'term',t,'steps',k+1,'pos',s[:3].tolist(),flush=True)
    env.close()

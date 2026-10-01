import numpy as np
from env_client import make_env
env=make_env()
def ray(seed, th, step=0.1, n=100):
    o,_=env.reset(seed=seed)
    d=np.array([np.cos(th),np.sin(th)])*step
    prev=o[:2].copy()
    for k in range(n):
        a=np.zeros(11); a[0],a[1]=d
        o,r,t,tr,_=env.step(a)
        if np.linalg.norm(o[:2]-prev)<1e-6: break
        prev=o[:2].copy()
    return prev
for seed in [0,1,2]:
    out=[]
    for ang in range(0,360,15):
        p=ray(seed, np.radians(ang))
        out.append((ang, round(float(p[0]),2), round(float(p[1]),2)))
    print(seed, out)

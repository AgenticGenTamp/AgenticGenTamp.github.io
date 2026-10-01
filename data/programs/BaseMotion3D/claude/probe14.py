import numpy as np
from env_client import make_env
env=make_env()
def run(seed, joints, gx=None, gy=None, rot=0.0, tolscan=False):
    o,_=env.reset(seed=seed)
    tx,ty=float(o[19]),float(o[20])
    if gx is None: gx,gy=tx,ty
    term=False
    for k in range(60):
        a=np.zeros(11)
        a[0]=np.clip(gx-o[0],-0.4,0.4); a[1]=np.clip(gy-o[1],-0.4,0.4); a[2]=np.clip(rot-o[2],-0.4,0.4)
        for j in range(7): a[3+j]=np.clip(joints[j]-o[3+j],-0.4,0.4)
        o,r,t,tr,_=env.step(a)
        if t: return True,k
        if np.all(np.abs(a[:10])<1e-7): return False,k
    return False,-1
base=[0,-0.35,-3.1416,-2.5,0,-0.87,1.5708]
print("default at target:", run(0, base))
for j in range(7):
    q=list(base); q[j]+=1.0
    print("joint",j,"+1.0 ->", run(0,q))
for j in range(7):
    q=list(base); q[j]-=1.0
    print("joint",j,"-1.0 ->", run(0,q))

import numpy as np
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
# target distribution
e=make_env()
ts=[]
for s in range(30):
    o,_=e.reset(seed=s); ts.append(o[19:22].copy())
ts=np.array(ts); print(ts.min(0), ts.max(0))
print("radius range", np.hypot(ts[:,0],ts[:,1]).min(), np.hypot(ts[:,0],ts[:,1]).max())
# tolerance test: approach along x axis only, stop short by d
def test(d):
    env=make_env(); o,_=env.reset(seed=0)
    tx,ty=o[19],o[20]
    r=np.hypot(tx,ty); ux,uy=tx/r,ty/r
    gx,gy=tx-ux*d, ty-uy*d
    for i in range(20):
        a=np.zeros(11)
        a[0]=np.clip(gx-o[0],-0.4,0.4); a[1]=np.clip(gy-o[1],-0.4,0.4)
        o,rr,t,tr,_=env.step(a)
        if t: return True,i
        if abs(o[0]-gx)<1e-6 and abs(o[1]-gy)<1e-6: 
            # one more step
            o,rr,t,tr,_=env.step(np.zeros(11))
            return t,i
    return False,-1
for d in [0.0,0.05,0.1,0.15,0.2,0.3,0.4,0.5]:
    print(d, test(d))

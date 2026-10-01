import numpy as np
from env_client import make_env
env=make_env()
same=True
for s in [0,7,42,99,333,777,1234,5000]:
    o,_=env.reset(seed=s)
    if not np.allclose(o[:19], np.array([0,0,0,0,-0.35,-3.1416,-2.5,0,-0.87,1.5708,0,0,0,0,0,0,0,0,0]),atol=1e-4):
        print("DIFFERENT start", s, o[:19]); same=False
print("start identical:", same)
# wall check on several seeds
def miny(seed,x=0.3):
    o,_=env.reset(seed=seed)
    for k in range(20):
        a=np.zeros(11); a[0]=np.clip(x-o[0],-0.4,0.4); a[1]=np.clip(-1.7-o[1],-0.4,0.4)
        if np.all(np.abs(a)<1e-7): break
        o,r,t,tr,_=env.step(a)
    y=float(o[1]); f=0.02
    while f>0.0005:
        a=np.zeros(11); a[1]=-f
        o2,r,t,tr,_=env.step(a)
        if abs(float(o2[1])-(y-f))<1e-6: o=o2;y=float(o[1])
        else: f/=2
    return round(y,4)
print([miny(s) for s in [0,7,42,99,333,777,1234,5000]])

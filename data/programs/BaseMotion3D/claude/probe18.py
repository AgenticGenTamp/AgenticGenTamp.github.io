import numpy as np
from env_client import make_env
env=make_env()
def miny(x, ystart=-1.7, seed=0, fine=0.005, ylim=-3.0):
    o,_=env.reset(seed=seed)
    for k in range(40):
        a=np.zeros(11); a[0]=np.clip(x-o[0],-0.4,0.4); a[1]=np.clip(ystart-o[1],-0.4,0.4)
        if np.all(np.abs(a)<1e-7): break
        o,r,t,tr,_=env.step(a)
    y=float(o[1])
    while y>ylim:
        a=np.zeros(11); a[1]=-fine
        o2,r,t,tr,_=env.step(a)
        if abs(float(o2[1])-y+fine)>1e-6: break
        o=o2; y=float(o[1])
    return y
for x in [-1.2,-1.0,-0.95,-0.9,-0.85,-0.82,-0.8,-0.75,-0.7,0.0,1.3,1.4,1.45,1.5,1.55,1.6,1.7]:
    print(round(x,3), round(miny(x),4), flush=True)

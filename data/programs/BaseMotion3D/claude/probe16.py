import numpy as np
from env_client import make_env
env=make_env()
def miny(rot, x=0.0):
    o,_=env.reset(seed=0)
    for k in range(30):
        a=np.zeros(11); a[2]=np.clip(rot-o[2],-0.4,0.4); a[0]=np.clip(x-o[0],-0.4,0.4); a[1]=np.clip(-1.5-o[1],-0.4,0.4)
        if np.all(np.abs(a)<1e-7): break
        o,r,t,tr,_=env.step(a)
    y=float(o[1])
    for k in range(60):
        a=np.zeros(11); a[1]=-0.01
        o2,r,t,tr,_=env.step(a)
        if abs(float(o2[1])-y)<1e-9: break
        o=o2; y=float(o[1])
    return round(y,3), round(float(o[2]),2)
for deg in [0,30,45,60,90,120,135,180,225,270]:
    print(deg, miny(np.radians(deg)))

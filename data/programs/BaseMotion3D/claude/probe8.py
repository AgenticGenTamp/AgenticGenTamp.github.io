import numpy as np
from env_client import make_env
env=make_env()
def limit(x, rot=0.0):
    o,_=env.reset(seed=0)
    for k in range(40):
        a=np.zeros(11); a[0]=np.clip(x-o[0],-0.4,0.4); a[2]=np.clip(rot-o[2],-0.4,0.4)
        o,r,t,tr,_=env.step(a)
        if abs(o[0]-x)<1e-6 and abs(o[2]-rot)<1e-6: break
    for k in range(40):
        a=np.zeros(11); a[1]=-0.4
        o,r,t,tr,_=env.step(a)
    return float(o[1]), float(o[0])
for x in [-3,-1,0,1,2.5]:
    print("x",x,limit(x))
print("rot pi:", limit(0.0, np.pi))
print("rot pi/2:", limit(0.0, np.pi/2))
# also max y
o,_=env.reset(seed=0)
for k in range(40):
    o,r,t,tr,_=env.step(np.array([0,0.4,0,0,0,0,0,0,0,0,0.0]))
print("maxy",o[1])

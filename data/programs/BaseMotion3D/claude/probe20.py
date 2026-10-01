import numpy as np
from env_client import make_env
env=make_env()
JLO=np.array([-16,-2.41,-19,-2.5,-16,-0.87,-14.4]); JHI=np.array([16,0.45,12.8,2.66,16,2.23,17.5])
base=np.array([0,-0.35,-3.1416,-2.5,0,-0.87,1.5708])
def miny(joints, rot=0.0, x=0.0):
    o,_=env.reset(seed=0)
    joints=np.clip(joints,JLO,JHI)
    for k in range(60):
        a=np.zeros(11)
        a[2]=np.clip(rot-o[2],-0.4,0.4)
        for j in range(7): a[3+j]=np.clip(joints[j]-o[3+j],-0.4,0.4)
        if np.all(np.abs(a)<1e-7): break
        o,r,t,tr,_=env.step(a)
    for k in range(20):
        a=np.zeros(11); a[0]=np.clip(x-o[0],-0.4,0.4); a[1]=np.clip(-1.7-o[1],-0.4,0.4)
        if np.all(np.abs(a)<1e-7): break
        o,r,t,tr,_=env.step(a)
    y=float(o[1]); fine=0.02
    while fine>0.0005:
        a=np.zeros(11); a[1]=-fine
        o2,r,t,tr,_=env.step(a)
        if abs(float(o2[1])-(y-fine))<1e-6: o=o2; y=float(o[1])
        else: fine/=2
    return round(y,4)
print("default", miny(base))
tests={"zeros":np.zeros(7),"j2=0.45":base+np.array([0,0.8,0,0,0,0,0]),
 "j2=-2.41":np.array([0,-2.41,-3.14,-2.5,0,-0.87,1.57]),
 "j4=0":np.array([0,-0.35,-3.14,0,0,-0.87,1.57]),
 "j4=2.6":np.array([0,-0.35,-3.14,2.6,0,-0.87,1.57]),
 "j6=2.2":np.array([0,-0.35,-3.14,-2.5,0,2.2,1.57]),
 "j3=0":np.array([0,-0.35,0,-2.5,0,-0.87,1.57]),
 "j1=1.57":np.array([1.57,-0.35,-3.14,-2.5,0,-0.87,1.57]),
 "j1=3.14":np.array([3.14,-0.35,-3.14,-2.5,0,-0.87,1.57]),
 "straight":np.array([0,0,0,0,0,0,0]),
}
for n,q in tests.items(): print(n, miny(q), flush=True)

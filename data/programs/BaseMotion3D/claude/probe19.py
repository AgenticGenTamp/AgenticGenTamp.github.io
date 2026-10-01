import numpy as np
from env_client import make_env
env=make_env()
JLO=np.array([-16,-2.41,-19,-2.5,-16,-0.87,-14.4]); JHI=np.array([16,0.45,12.8,2.66,16,2.23,17.5])
def term(seed, joints, gx, gy):
    o,_=env.reset(seed=seed)
    joints=np.clip(joints,JLO,JHI)
    for k in range(40):
        a=np.zeros(11)
        for j in range(7): a[3+j]=np.clip(joints[j]-o[3+j],-0.4,0.4)
        if np.all(np.abs(a)<1e-7): break
        o,r,t,tr,_=env.step(a)
        if t: return True
    for k in range(40):
        a=np.zeros(11); a[0]=np.clip(gx-o[0],-0.4,0.4); a[1]=np.clip(gy-o[1],-0.4,0.4)
        if np.all(np.abs(a)<1e-7): return False
        o,r,t,tr,_=env.step(a)
        if t: return True
    return False
def radius(joints, ang, seed=0):
    o,_=env.reset(seed=seed); tx,ty=float(o[19]),float(o[20])
    lo,hi=0.0,0.6
    for _ in range(12):
        mid=(lo+hi)/2
        if term(seed,joints,tx-mid*np.cos(ang),ty-mid*np.sin(ang)): lo=mid
        else: hi=mid
    return round(lo,3)
base=np.array([0,-0.35,-3.1416,-2.5,0,-0.87,1.5708])
configs={"default":base, "zeros":np.zeros(7), "j2max":base+np.array([0,0.8,0,0,0,0,0]),
         "j4max":base+np.array([0,0,0,5,0,0,0]), "j6":base+np.array([0,0,0,0,0,3,0]),
         "mix":np.array([1.0,0.4,-2.0,-0.5,1.0,1.5,0.0])}
for name,q in configs.items():
    print(name, [radius(q,np.radians(a)) for a in (0,90,180,270)], flush=True)

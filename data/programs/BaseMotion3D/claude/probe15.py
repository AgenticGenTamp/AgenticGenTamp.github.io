import numpy as np
from env_client import make_env
env=make_env()
JLO=np.array([-16,-2.41,-19,-2.5,-16,-0.87,-14.4])
JHI=np.array([16,0.45,12.8,2.66,16,2.23,17.5])
def run(seed, joints, gx=None, gy=None, rot=0.0):
    o,_=env.reset(seed=seed)
    tx,ty=float(o[19]),float(o[20])
    if gx is None: gx,gy=tx,ty
    joints=np.clip(joints,JLO,JHI)
    # phase 1: set joints and rot, base stays
    for k in range(30):
        a=np.zeros(11)
        a[2]=np.clip(rot-o[2],-0.4,0.4)
        for j in range(7): a[3+j]=np.clip(joints[j]-o[3+j],-0.4,0.4)
        if np.all(np.abs(a)<1e-7): break
        o,r,t,tr,_=env.step(a)
        if t: return "term_phase1",k
    # phase 2 move base
    for k in range(40):
        a=np.zeros(11)
        a[0]=np.clip(gx-o[0],-0.4,0.4); a[1]=np.clip(gy-o[1],-0.4,0.4)
        if np.all(np.abs(a)<1e-7): 
            return False, float(o[0]),float(o[1])
        o,r,t,tr,_=env.step(a)
        if t: return True,k
    return False,-1
base=np.array([0,-0.35,-3.1416,-2.5,0,-0.87,1.5708])
print("default:", run(0,base))
for j in range(7):
    for d in (1.0,-1.0):
        q=base.copy(); q[j]+=d
        print("joint",j,d,"->", run(0,q))

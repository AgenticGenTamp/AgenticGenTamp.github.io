import sys; sys.path.insert(0,'.')
import numpy as np, math, time
from approach import *
from kin import ik_down, Q0, fk_full, LO, HI, shoulder_xy
rng=np.random.default_rng(0)
def init_for(base,p):
    sh=shoulder_xy(base); a=math.atan2(p[1]-sh[1],p[0]-sh[0])-base[2]
    q=Q0.copy(); q[0]=np.clip(wrap(a)+0.0,LO[0],HI[0]); return q
tot=0; s=[0,0,0]; T=[0,0,0]
for trial in range(300):
    x=rng.uniform(-0.26,0.26); y=rng.uniform(-0.56,0.56); yaw=rng.uniform(-math.pi,math.pi)
    bl=[bb for bb in BASE_CANDS+block_bases(x,y) if 0.25<np.hypot(*(np.array(shoulder_xy(bb))-[x,y]))<0.85]
    for base in bl[::3]:
        tot+=1; p=np.array([x,y,0.793])
        t=time.time(); q,ok=ik_down(base,p,Q0,yaw=yaw,iters=250); T[0]+=time.time()-t; s[0]+=ok
        t=time.time(); q,ok2=ik_down(base,p,init_for(base,p),yaw=yaw,iters=250); T[1]+=time.time()-t; s[1]+=ok2
        s[2]+= (ok or ok2)
print(tot,s,[x/tot*1000 for x in T])
rng=np.random.default_rng(5); fails=0; solv=0; firsts=[]
for trial in range(120):
    x=rng.uniform(-0.26,0.26); y=rng.uniform(-0.56,0.56); yaw=rng.uniform(-math.pi,math.pi)
    bl=[bb for bb in BASE_CANDS+block_bases(x,y) if 0.25<np.hypot(*(np.array(shoulder_xy(bb))-[x,y]))<0.85]
    for base in bl[::3]:
        p=np.array([x,y,0.793])
        q,ok=ik_down(base,p,Q0,yaw=yaw,iters=250)
        if ok: continue
        q,ok=ik_down(base,p,init_for(base,p),yaw=yaw,iters=250)
        if ok: continue
        fails+=1
        for t in range(20):
            qi=rng.uniform(np.maximum(LO,-3),np.minimum(HI,3))
            q,ok=ik_down(base,p,qi,yaw=yaw,iters=250)
            if ok: solv+=1; firsts.append(t); print(np.round(np.r_[base, np.hypot(*(np.array(shoulder_xy(base))-[x,y]))],2), np.round(q,2)); break
print("fails",fails,"solvable",solv, firsts)

import sys; sys.path.insert(0,'.')
import numpy as np, math, time
from approach import *
from kin import ik_down, Q0, fk_full, LO, HI
rng=np.random.default_rng(0)
# statistics: random block positions & bases from block_bases; success rate of Q0-init vs multi-init
tot=0; s1=0; s2=0; s3=0; t1=t2=0
for trial in range(150):
    x=rng.uniform(-0.26,0.26); y=rng.uniform(-0.56,0.56); yaw=rng.uniform(-math.pi,math.pi)
    for base in block_bases(x,y)[:6]:
        tot+=1
        p=np.array([x,y,0.793])
        t=time.time(); q,ok=ik_down(base,p,Q0,yaw=yaw); t1+=time.time()-t
        s1+=ok
        t=time.time(); q,ok2=ik_down(base,p,Q0,yaw=yaw,iters=250); t2+=time.time()-t
        s2+=ok2
        if not ok2:
            Q1=Q0.copy(); Q1[2]=-Q0[2]+1.0; Q1[4]=Q0[4]-math.pi/2
            for Qi in [np.array([0.,0.,0.,-1.2,0.,-1.0,0.]), np.array([0.5,0.2,1.5,-1.5,1.5,-1.5,2.0]), np.array([-0.3,0.2,-0.5,-1.5,-1.5,-1.5,0.])]:
                q,ok3=ik_down(base,p,Qi,yaw=yaw,iters=150)
                if ok3: break
            s3+=ok3
print(tot, s1, s2, s2+s3, t1/tot*1000, t2/tot*1000)

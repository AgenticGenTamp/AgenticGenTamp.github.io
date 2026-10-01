import sys; sys.path.insert(0,'.')
import numpy as np, time, math
from kin import *
from kin import _fk_fast
rng=np.random.default_rng(0)
for i in range(5):
    b=rng.uniform(-1,1,3); q=rng.uniform(-2,2,7)
    _,ax,orr,t,R=fk_full(b,q); a2,o2,t2,cols=_fk_fast(*b,q)
    print(np.abs(np.array(a2)-ax).max(), np.abs(np.array(o2)-orr).max(), np.abs(np.array(cols).T-R).max())
same=0; n=0; t1=t2=0
for i in range(300):
    base=(rng.uniform(-0.7,-0.45),rng.uniform(-0.2,0.2),0.0)
    pos=np.array([rng.uniform(-0.2,0.2),rng.uniform(-0.4,0.4),rng.uniform(0.79,0.9)])
    yaw=rng.uniform(-3,3) if i%2 else None
    po=None if i%3 else np.array([0.012,0.003,0.0])
    s=time.time(); qa,oka=ik_down(base,pos,Q0,yaw=yaw,p_off=po); t1+=time.time()-s
    s=time.time(); qb,okb=ik_down_fast(base,pos,Q0,yaw=yaw,p_off=po); t2+=time.time()-s
    n+=1; same+= (oka==okb) and (not oka or np.abs(wrap(qa-qb)).max()<1e-4)
print(same,n,t1/n*1e3,t2/n*1e3)

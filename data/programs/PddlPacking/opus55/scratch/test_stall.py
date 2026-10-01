import sys; sys.path.insert(0,'.')
import numpy as np, math, time
from approach import *
from kin import ik_down_fast, Q0
rng=np.random.default_rng(0)
res={True:[0,0,0],False:[0,0,0]}
for trial in range(250):
    x=rng.uniform(-0.26,0.26); y=rng.uniform(-0.56,0.56); yaw=rng.uniform(-math.pi,math.pi)
    for base in (BASE_CANDS+block_bases(x,y))[::4]:
        p=np.array([x,y,0.793])
        for st in (True,False):
            t=time.time(); q,ok=ik_down_fast(base,p,Q0,yaw=yaw,iters=250,stall=st); res[st][1]+=time.time()-t
            res[st][0]+=ok; res[st][2]+=1
print(res)

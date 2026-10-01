import sys; sys.path.insert(0,'.')
import numpy as np, math, time
import kin
from approach import *
from kin import ik_down_fast, Q0
for P in ([40,1e-3,0.9,30],[30,1e-2,0.9,20],[40,4e-3,0.95,30],[60,1e-3,0.95,40]):
    kin.STALL[:]=P
    rng=np.random.default_rng(0); ok_n=0; T=0
    for trial in range(250):
        x=rng.uniform(-0.26,0.26); y=rng.uniform(-0.56,0.56); yaw=rng.uniform(-math.pi,math.pi)
        for base in (BASE_CANDS+block_bases(x,y))[::4]:
            p=np.array([x,y,0.793])
            t=time.time(); q,ok=ik_down_fast(base,p,Q0,yaw=yaw,iters=250); T+=time.time()-t; ok_n+=ok
    print(P, ok_n, T)

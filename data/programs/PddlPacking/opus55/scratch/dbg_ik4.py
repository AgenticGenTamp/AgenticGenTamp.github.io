import sys; sys.path.insert(0,'.')
import numpy as np, math
from approach import *
from kin import ik_down, Q0, fk_full, LO, HI
b=np.array([0.243,-0.472,0.781,2.377]); p=np.array([b[0],b[1],0.793])
rng=np.random.default_rng(1)
for base in [(0.0,-0.99,0.0),(0.15,-0.99,0.0),(0.3,-0.99,0.0)]:
    n=0
    for t in range(30):
        qi=rng.uniform(np.maximum(LO,-3),np.minimum(HI,3))
        q,ok=ik_down(base,p,qi,yaw=b[3],iters=300)
        if ok:
            n+=1
            if n==1: print(base, np.round(q,2))
    print(base,"succ",n)

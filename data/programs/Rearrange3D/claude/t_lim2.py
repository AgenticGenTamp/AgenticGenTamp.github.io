import numpy as np
from env_client import make_env
env=make_env()
# Verify clamps for joints 2,4,6 are hard joint limits (not collisions):
# reset arm elsewhere first (move joint1 by 1.5 rad) then push joint2/4/6 to extremes
mv=np.zeros(11,dtype=np.float32); mv[3]=0.1
for j,cmax in [(1,1.2778),(3,2.5833),(5,2.1091)]:
    for sgn in [1,-1]:
        obs,_=env.reset(seed=0)
        for _ in range(60): obs,_,_,_,_=env.step(mv)   # rotate joint1 +1.5 rad
        a=np.zeros(11,dtype=np.float32); a[3+j]=0.1*sgn
        prev=None
        for t in range(300):
            obs,_,_,_,_=env.step(a); cur=float(np.asarray(obs)[96+j])
            if prev is not None and abs(cur-prev)<1e-6: break
            prev=cur
        print(f"joint{j+1} sgn{sgn:+d} stops at {cur:.4f} after {t} steps")
env.close()

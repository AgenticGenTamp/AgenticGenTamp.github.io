import numpy as np
from kin import ik, down_R, fk
from th import H
h=H(312,oc=5); base=h.base(); print('base',base, 'q',h.q().round(2))
rng=np.random.default_rng(0)
JL={1:2.41,3:2.66,5:2.23}
for pos in [(0.106,0.006,0.146),(0.13,0.005,0.153),(0.106,0.006,0.2),(0.2,0.0,0.146),(0.12,0.1,0.146)]:
    best=None
    for t in range(100):
        q0=rng.uniform(-2.2,2.2,7)
        for yaw in [0, np.pi/2]:
            q,e=ik(base,q0,np.array(pos),down_R(yaw),iters=200)
            ok=all(abs(q[i])<=JL[i] for i in JL)
            if ok and (best is None or e<best[0]): best=(e,yaw,q.round(2))
    print(pos,best)

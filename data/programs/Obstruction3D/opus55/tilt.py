import numpy as np
from kin import ik, down_R, fk
from th import H
def Ry(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
h=H(312,oc=5); base=h.base()
rng=np.random.default_rng(0)
JL={1:2.41,3:2.66,5:2.23}
pos=np.array((0.106,0.006,0.146))
for tilt in [0.2,0.4,0.6]:
    for yaw in [0,np.pi/2]:
        R=Ry(tilt)@down_R(yaw)   # world-frame rotation about y: tool z axis leans
        best=None
        for t in range(60):
            q0=rng.uniform(-2.2,2.2,7)
            q,e=ik(base,q0,pos,R,iters=200)
            ok=all(abs(q[i])<=JL[i] for i in JL)
            if ok and (best is None or e<best[0]): best=(e,q)
        print(tilt,yaw,None if best is None else best[0], fk(base,best[1])[:3,2].round(2) if best else '')

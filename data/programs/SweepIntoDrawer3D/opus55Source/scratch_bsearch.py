import numpy as np
from approach import *
from approach import _get_model
k=Kin(_get_model())
q0=np.array([0.,-0.349,3.142,-2.548,0.,-0.873,1.571])
rng=np.random.default_rng(0)
res=[]
for n in range(1500):
    bx=rng.uniform(1.2,1.5); by=rng.uniform(-0.6,0.6); yaw=np.pi+rng.uniform(-1.3,1.3); psi=rng.uniform(-0.2,0.9)
    cs=[bx+np.cos(yaw)*0.29-np.sin(yaw)*s for s in (0.27,-0.27)]
    if min(cs)<0.97: continue
    q,ok=k.ik(np.array([bx,by,yaw]),q0,np.array([0.934+0.09,0,0.355]),R_front(psi),iters=80)
    if not ok: continue
    for j in (0,2,4,6): q[j]=q0[j]+((q[j]-q0[j]+np.pi)%(2*np.pi)-np.pi)
    if np.any(q<LIM[:,0]) or np.any(q>LIM[:,1]): continue
    res.append((np.abs(q-q0).max(),bx,by,yaw,psi))
res.sort()
for r in res[:10]: print(np.round(r,3))

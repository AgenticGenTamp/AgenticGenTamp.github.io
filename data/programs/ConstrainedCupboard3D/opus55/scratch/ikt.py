import sys; sys.path.insert(0,'/sandbox'); sys.path.insert(0,'/sandbox/scratch')
import numpy as np
from kin import ik, fk, fk_all
from ctrl import TOOL
RH = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0.]])
RH2 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.]])
home=np.array([0, 0.26, np.pi, -2.27, 0, 0.96, np.pi/2])
rng=np.random.default_rng(0)
def ok(q):
    fr,T=fk_all(q,TOOL)
    return all(f[2,3]>-0.3 for f in fr) and all(f[0,3]>-0.1 for f in fr[2:])
for R,nm in [(RH,'RH'),(RH2,'RH2')]:
  for pa in [[0.45,0,-0.25],[0.55,0,-0.25],[0.65,0,-0.25],[0.55,0,0.0],[0.65,0,0.1],[0.75,0,0.0],[0.55,0,-0.35]]:
    best=None
    for i in range(150):
        s0=home+rng.normal(0,1.0,7)
        q,e1,e2=ik(s0,np.array(pa),R,tool=TOOL,iters=300)
        if e1<1e-3 and e2<1e-2 and ok(q):
            d=q-home; d[[0,2,4,6]]=(d[[0,2,4,6]]+np.pi)%(2*np.pi)-np.pi; c=np.abs(d).sum()
            if best is None or c<best[0]: best=(c,q)
    print(nm,pa, None if best is None else (round(best[0],2),best[1].round(2)))

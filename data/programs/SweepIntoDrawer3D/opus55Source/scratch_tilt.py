import numpy as np, sys
sys.path.insert(0,'.')
from approach import R_grasp
from kin import make_model, Kin, Q_HOME
env, model = make_model()
kin = Kin(model)
b=np.array([1.5,0,np.pi])
for tilt in (0.0,0.1,0.2):
  for r in (0.62,0.70,0.78):
    qs={}
    q=None
    seed=Q_HOME
    for phi in np.linspace(np.pi/2-0.8,np.pi/2+0.8,5):
        q = kin.ik_multi(b, seed, np.array([1.5-r,0,0.55]), R_grasp(tilt,phi))
        if q is None: qs[round(phi,2)]=None; continue
        seed=q; qs[round(phi,2)]=q
    ks=list(qs)
    if any(v is None for v in qs.values()): print(tilt,r,"fail",[k for k in ks if qs[k] is None]); continue
    d=[np.round(np.abs(qs[ks[k+1]]-qs[ks[k]]),2) for k in range(len(ks)-1)]
    print(tilt, r, "per 0.4rad:", [tuple(x) for x in d[:2]])

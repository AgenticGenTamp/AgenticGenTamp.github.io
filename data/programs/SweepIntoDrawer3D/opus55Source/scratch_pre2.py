import numpy as np, sys
sys.path.insert(0,'.')
from approach import R_hook
from kin import make_model, Kin, Q_RETRACT
env, model = make_model()
kin = Kin(model)
q0 = Q_RETRACT
res=[]
for a in (0.3,0.4,0.5,0.6):
  R = R_hook(a); dv=R[:,2]
  tip=np.array([0.912,0,0.322]); site = tip - 0.02*dv
  for by in np.arange(-0.4,0.41,0.05):
    for yaw in np.pi+np.arange(-0.6,0.61,0.15):
      for bx in (1.38,1.42,1.46):
        b=np.array([bx,by,yaw])
        q = kin.ik_multi(b, q0, site+[0,0,0.08], R)
        if q is None: continue
        q2 = kin.ik_multi(b, q, site, R)
        if q2 is None: continue
        # pulled config: base +0.12 x with same q must be ok (it is rigid); check q2 limits fine
        res.append((np.max(np.abs(q-q0)), a, by, yaw-np.pi, bx, np.round(q,2)))
res.sort(key=lambda r:r[0])
for r in res[:15]: print(round(r[0],2), r[1], round(r[2],2), round(r[3],2), r[4], r[5])

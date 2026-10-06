import numpy as np, sys
sys.path.insert(0,'.')
from approach import R_hook
from kin import make_model, Kin, Q_RETRACT, LIM
env, model = make_model()
kin = Kin(model)
q0 = Q_RETRACT
rng=np.random.default_rng(0)
a=0.5; R=R_hook(a); dv=R[:,2]
site=np.array([0.912,0,0.322])-0.02*dv
b=np.array([1.42,-0.33,np.pi])
res=[]
for k in range(400):
    s = q0 + rng.uniform(-1.5,1.5,7)
    q,ok,e1,e2 = kin.ik(b, s, site+[0,0,0.08], R, iters=150)
    if not ok: continue
    for j in (0,2,4,6):
        q[j] = q0[j] + ((q[j]-q0[j]+np.pi)%(2*np.pi)-np.pi)
    q2,ok2,_,_ = kin.ik(b, q, site, R, iters=150)
    if not ok2 or np.max(np.abs(q2-q))>0.3: continue
    res.append((np.max(np.abs(q-q0)), np.round(q,3)))
res.sort(key=lambda r:r[0])
for r in res[:8]: print(round(r[0],2), list(r[1]))

import numpy as np, approach, mujoco
from approach import Kin, down_rot, HOME, wrap
k=Kin(); rng=np.random.default_rng(0)
best=[]
for trial in range(400):
    sd = HOME + rng.normal(0,1.0,7)
    r=rng.uniform(0.3,0.7); lat=rng.uniform(-0.4,0.4); psi=rng.uniform(-np.pi,np.pi)
    q,pe,re_=k.ik(np.zeros(3), np.array([r,lat,0.015]), down_rot(psi), sd, iters=200)
    if pe<2e-3 and re_<5e-3:
        c=np.max(np.abs([wrap(x) for x in q-HOME]))
        best.append((c, r, lat, psi, np.round(q,2)))
best.sort(key=lambda x:x[0])
for b in best[:8]: print(round(b[0],2), round(b[1],2), round(b[2],2), round(b[3],2), b[4])

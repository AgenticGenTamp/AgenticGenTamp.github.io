import numpy as np
from approach import _Kin, Q_PICK_SEED, Q_PLACE_SEED, GeneratedApproach
k=_Kin()
ap=GeneratedApproach.__new__(GeneratedApproach); ap.kin=k; ap._cfg_cache={}
cfg=ap._configs(0.004)
qa,qb,qh=cfg['qa'],cfg['qb'],cfg['qh']
for t in np.linspace(0,1,11):
    q=qb+(qa-qb)*t
    c,R=k.cube_center(q)
    print(round(t,1), c.round(3))
print("qh-qb",np.abs(qh-qb).max(),"qa-qb",np.abs(qa-qb).max())

import numpy as np
from helper import Sim
def probe(s,d,steps=(0.005,0.001,0.0002,0.00005),maxn=900):
    d=np.array(d,float)
    for st in steps:
        for i in range(maxn):
            p=s.obs[:2].copy(); s.step([d[0]*st,d[1]*st,0,0,0.0])
            if np.linalg.norm(s.obs[:2]-p)<1e-12: break
    return s.obs[:2].copy()
for seed in [0,1,2,3,7]:
    s=Sim(seed); o=s.obs.copy()
    print(f"seed {seed} hook C=({o[9]:.3f},{o[10]:.3f}) th={o[11]:.3f} rob=({o[0]:.2f},{o[1]:.2f}) mov=({o[20]:.3f},{o[21]:.3f}) tgt=({o[29]:.3f},{o[30]:.3f})")
    for arm,th in [(0.1,0.0),(0.2,np.pi),(0.2,0.0)]:
        s2=Sim(seed); r=s2.goto(3.0,0.3,th,arm,0.0)
        p=probe(s2,(-1,0)); print(f"   -x arm={arm} th={th:.2f} xmin={p[0]:.5f}")
        s2.close()
    s.close()

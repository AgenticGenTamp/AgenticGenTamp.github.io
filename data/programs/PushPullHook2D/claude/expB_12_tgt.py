import numpy as np
from helper import Sim
def probe(s,d,vac,steps=(0.01,0.002,0.0005,0.0001),maxn=900):
    d=np.array(d,float)
    for st in steps:
        for i in range(maxn):
            p=s.obs[:2].copy(); s.step([d[0]*st,d[1]*st,0,0,vac])
            if np.linalg.norm(s.obs[:2]-p)<1e-12: break
    return s.obs.copy()
for seed,side in [(6,-1),(6,1),(18,-1)]:
    s=Sim(seed); T=s.obs[29:31].copy(); M=s.obs[20:22].copy()
    print(f"seed{seed} T={np.round(T,4)} M={np.round(M,4)} hookC={np.round(s.obs[9:12],3)}")
    s.goto(T[0]-side*0.6,0.5,np.pi/2,0.1,0.0); s.goto(T[0]-side*0.6,1.15,np.pi/2,0.2,0.0)
    o=probe(s,(side,0),0.0)
    gc=np.array([o[0]+side*0.035,o[1]+0.195])
    print(f"  approach {'+x' if side>0 else '-x'}: rob=({o[0]:.4f},{o[1]:.4f}) distT={np.hypot(*(T-gc)):.4f} distM={np.hypot(*(M-gc)):.4f} Tnow={np.round(o[29:31],4)}")
    s.close()

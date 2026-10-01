import numpy as np
from helper import Sim
def probe(s,d,vac,steps=(0.01,0.002,0.0005,0.0001),maxn=900):
    d=np.array(d,float)
    for st in steps:
        for i in range(maxn):
            p=s.obs[:2].copy(); s.step([d[0]*st,d[1]*st,0,0,vac])
            if np.linalg.norm(s.obs[:2]-p)<1e-12: break
    return s.obs.copy()
s=Sim(15)
print("M",np.round(s.obs[20:22],4))
print("rot",s.goto(1.9,0.5,np.pi/2,0.1,0.0), np.round(s.obs[[0,1,2,4]],4))
print("arm",s.goto(1.9,1.15,np.pi/2,0.2,0.0), np.round(s.obs[[0,1,2,4]],4))
o=probe(s,(1,0),0.0)
print(f"+x grip-up: rob=({o[0]:.4f},{o[1]:.4f},{o[2]:.3f}) tip=({o[0]:.4f},{o[1]+0.205:.4f}) gripRt x={o[0]+0.035:.4f} M=({o[20]:.4f},{o[21]:.4f})")
print("  dist(gripcorner,M)=",round(np.hypot(o[20]-(o[0]+0.035),o[21]-(o[1]+0.195)),4), " M-r bottom=",round(o[21]-0.05,4))
s.close()
# ymax across seeds
for seed in [3,10,15]:
    s=Sim(seed); s.goto(1.2,0.6,0.0,0.1,0.0); o=probe(s,(0,1),0.0)
    print(f"seed {seed} ymax at x=1.2 arm0.1: {o[1]:.5f}")
    s.close()

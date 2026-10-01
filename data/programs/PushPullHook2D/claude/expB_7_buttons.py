import numpy as np
from helper import Sim
def hk(o):
    C=o[9:11]; th=o[11]; a=np.array([-np.cos(th),-np.sin(th)]); b=np.array([np.sin(th),-np.cos(th)])
    return C, C+a*o[18], C+b*o[19]
def probe(s,d,vac,steps=(0.01,0.002,0.0005,0.0001),maxn=900):
    d=np.array(d,float)
    for st in steps:
        for i in range(maxn):
            p=s.obs[:2].copy(); s.step([d[0]*st,d[1]*st,0,0,vac])
            if np.linalg.norm(s.obs[:2]-p)<1e-12: break
    return s.obs.copy()
def show(tag,o):
    C,L,S=hk(o)
    print(f"{tag}: rob=({o[0]:.4f},{o[1]:.4f},{o[2]:+.3f}) C=({C[0]:.4f},{C[1]:.4f}) L=({L[0]:.4f},{L[1]:.4f}) S=({S[0]:.4f},{S[1]:.4f}) M=({o[20]:.4f},{o[21]:.4f})")
o0=Sim(42); print("tgt",np.round(o0.obs[29:31],4),"mov",np.round(o0.obs[20:22],4)); o0.close()
# --- ceiling test: grasp near tip so corner goes high
for d in [1.24,0.30]:
    s=Sim(42); s.grasp_hook(d)
    r=s.goto(1.75,s.obs[1],s.obs[2],0.2,1.0)
    o=probe(s,(0,1),1.0); show(f"ceiling d={d} up",o); s.close()
# --- hook vs TARGET button: hook up, sweep +x below/at target height
s=Sim(42); s.grasp_hook(1.15); s.goto(1.4,1.15,s.obs[2],0.2,1.0)
show("pre-sweep",s.obs)
o=probe(s,(1,0),1.0); show("hookUP sweep +x (tgt at 1.948,1.660)",o); s.close()
# same but starting right of target sweeping -x
s=Sim(42); s.grasp_hook(1.15); s.goto(3.0,1.15,s.obs[2],0.2,1.0)
o=probe(s,(-1,0),1.0); show("hookUP sweep -x from x=3.0",o); s.close()

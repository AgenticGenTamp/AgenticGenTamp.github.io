import numpy as np
from helper import Sim
def probe(s,d,steps=(0.01,0.002,0.0004,0.0001),maxn=900):
    d=np.array(d,float)
    for st in steps:
        for i in range(maxn):
            p=s.obs[:2].copy(); s.step([d[0]*st,d[1]*st,0,0,1.0])
            if np.linalg.norm(s.obs[:2]-p)<1e-12: break
    return s.obs.copy()
def desc(o,tag):
    C=o[9:11]; th=o[11]
    a=np.array([-np.cos(th),-np.sin(th)]); b=np.array([np.sin(th),-np.cos(th)])
    L=C+a*o[18]; S=C+b*o[19]
    xs=[C[0],L[0],S[0]]; ys=[C[1],L[1],S[1]]
    print(f"{tag}: rob=({o[0]:.4f},{o[1]:.4f},th={o[2]:+.3f}) C=({C[0]:.3f},{C[1]:.3f}) Ltip=({L[0]:.3f},{L[1]:.3f}) Stip=({S[0]:.3f},{S[1]:.3f}) bbox x[{min(xs):.3f},{max(xs):.3f}] y[{min(ys):.3f},{max(ys):.3f}]")

def setup(seed=42, th=np.pi, pos=(1.75,0.6)):
    s=Sim(seed); s.grasp_hook(1.15)
    r1=s.goto(s.obs[0],s.obs[1],th,0.2,1.0)
    r2=s.goto(pos[0],pos[1],th,0.2,1.0)
    return s,(r1,r2)

for name,th,d in [("hookUP",np.pi,(0,1)),("hookDOWN",0.0,(0,-1)),
                  ("hookLEFT",-np.pi/2,(-1,0)),("hookRIGHT",np.pi/2,(1,0)),
                  ("hookUP-movX+",np.pi,(1,0)),("hookUP-movX-",np.pi,(-1,0)),
                  ("hookDOWN-movX+",0.0,(1,0)),("hookLEFT-movY+",-np.pi/2,(0,1)),
                  ("hookLEFT-movY-",-np.pi/2,(0,-1))]:
    s,rr=setup(th=th)
    if rr[0]<0 or rr[1]<0: print(name,"setup stuck",rr); desc(s.obs,name+"(stuck)"); s.close(); continue
    o=probe(s,d); desc(o,f"{name} mv{d}")
    s.close()

import numpy as np
from helper import Sim
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
def hk(o):
    C=o[9:11]; th=o[11]; a=np.array([-np.cos(th),-np.sin(th)]); b=np.array([np.sin(th),-np.cos(th)])
    return C, C+a*o[18], C+b*o[19]
def probe(s,d,vac=1.0,steps=(0.01,0.002,0.0005,0.0001),maxn=900):
    d=np.array(d,float)
    for st in steps:
        for i in range(maxn):
            p=s.obs[:2].copy(); s.step([d[0]*st,d[1]*st,0,0,vac])
            if np.linalg.norm(s.obs[:2]-p)<1e-12: break
    return s.obs.copy()
def rotprobe(s,sgn,step=0.02,maxn=400):
    for i in range(maxn):
        p=s.obs[2]
        s.step([0,0,sgn*step,0,1.0])
        if abs(wrap(s.obs[2]-p))<1e-12: break
    for i in range(60):
        p=s.obs[2]; s.step([0,0,sgn*0.002,0,1.0])
        if abs(wrap(s.obs[2]-p))<1e-12: break
    return s.obs.copy()
def show(tag,o):
    C,L,S=hk(o)
    print(f"{tag}: rob=({o[0]:.4f},{o[1]:.4f},{o[2]:+.4f}) hth={o[11]:+.4f} C=({C[0]:.4f},{C[1]:.4f}) L=({L[0]:.4f},{L[1]:.4f}) S=({S[0]:.4f},{S[1]:.4f})")
for pos in [(1.75,0.9),(1.75,0.3),(1.75,1.15)]:
    for sgn in [1,-1]:
        s=Sim(42); s.grasp_hook(1.15)
        r=s.goto(pos[0],pos[1],s.obs[2],0.2,1.0)
        if r<0: print("move stuck",pos); s.close(); continue
        o=rotprobe(s,sgn); show(f"pos={pos} rot{sgn:+d}",o); s.close()

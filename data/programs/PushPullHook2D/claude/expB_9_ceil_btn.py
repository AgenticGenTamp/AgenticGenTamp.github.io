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
# CEILING: grasp near tip, park hook away from button in x, then raise
for d,x in [(1.245,3.1),(1.245,0.55)]:
    s=Sim(42); s.grasp_hook(d)
    r=s.goto(x,s.obs[1],s.obs[2],0.2,1.0)
    o=probe(s,(0,1),1.0); show(f"CEIL d={d} x={x} (Cmax expect {o[1]+0.26+d:.2f})",o); s.close()
# BASE/GRIPPER vs MOVABLE BUTTON, seed 15 (my=1.312)
s=Sim(15); o=s.obs
print("seed15 M=",np.round(o[20:22],4)," T=",np.round(o[29:31],4)," hook C",np.round(o[9:12],3))
r=s.goto(o[20],0.6,np.pi/2,0.2,0.0)
print(" goto res",r,"pos",np.round(s.obs[:2],3))
o=probe(s,(0,1),0.0); print(f" base up under button: rob=({o[0]:.4f},{o[1]:.4f}) tip_y={o[1]+0.205:.4f} M=({o[20]:.4f},{o[21]:.4f})")
s.close()
# same with arm 0.1 and theta down (no gripper up)
s=Sim(15); o=s.obs; s.goto(o[20],0.6,-np.pi/2,0.1,0.0)
o=probe(s,(0,1),0.0); print(f" base up (arm .1, grip down): rob y={o[1]:.4f} M=({o[20]:.4f},{o[21]:.4f})")
s.close()
# lateral: drive base sideways at ymax into button column
s=Sim(15); o=s.obs; s.goto(o[20]-0.5,1.15,np.pi/2,0.2,0.0)
o=probe(s,(1,0),0.0); print(f" base +x at ymax grip-up: rob=({o[0]:.4f},{o[1]:.4f}) M=({o[20]:.4f},{o[21]:.4f})")
s.close()

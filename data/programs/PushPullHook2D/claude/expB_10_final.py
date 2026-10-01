import numpy as np
from helper import Sim
def probe(s,d,vac,steps=(0.01,0.002,0.0005,0.0001),maxn=900):
    d=np.array(d,float)
    for st in steps:
        for i in range(maxn):
            p=s.obs[:2].copy(); s.step([d[0]*st,d[1]*st,0,0,vac])
            if np.linalg.norm(s.obs[:2]-p)<1e-12: break
    return s.obs.copy()
# --- clean lateral gripper-vs-button test (seed15 M=(2.2858,1.3119))
s=Sim(15); o=s.obs
s.goto(1.9,1.15,np.pi/2,0.2,0.0); print("start",np.round(s.obs[[0,1,2,4]],4))
o=probe(s,(1,0),0.0)
print(f"lateral +x grip-up: rob=({o[0]:.4f},{o[1]:.4f}) griprect x[{o[0]-0.035:.4f},{o[0]+0.035:.4f}] y[{o[1]+0.195:.4f},{o[1]+0.205:.4f}] M=({o[20]:.4f},{o[21]:.4f}) dxM={o[20]-o[0]:.4f}")
s.close()
s=Sim(15); s.goto(2.6,1.15,np.pi/2,0.2,0.0); o=probe(s,(-1,0),0.0)
print(f"lateral -x grip-up: rob=({o[0]:.4f},{o[1]:.4f}) M=({o[20]:.4f},{o[21]:.4f}) dxM={o[20]-o[0]:.4f}")
s.close()
# --- base circle vs button: arm0.1 grip sideways, ymax
s=Sim(15); s.goto(1.9,1.15,0.0,0.1,0.0); o=probe(s,(1,0),0.0)
print(f"lateral +x arm0.1 th=0: rob=({o[0]:.4f},{o[1]:.4f}) M=({o[20]:.4f},{o[21]:.4f})")
s.close()
# --- hook up at tip grasp, away from movable button: can base reach ymax 1.15?
for x in [0.8,3.1]:
    s=Sim(42); s.grasp_hook(1.20)
    s.goto(x,s.obs[1],s.obs[2],0.2,1.0)
    o=probe(s,(0,1),1.0)
    C=o[9:11]; print(f"hookUP tipgrasp x={x}: rob=({o[0]:.4f},{o[1]:.4f}) C=({C[0]:.4f},{C[1]:.4f}) dC={C[1]-o[1]:.4f} M=({o[20]:.4f},{o[21]:.4f})")
    s.close()
# --- push movable button up as high as possible (seed 42) to find its ceiling
s=Sim(42); s.grasp_hook(1.20); s.goto(1.5,s.obs[1],s.obs[2],0.2,1.0)
o=probe(s,(0,1),1.0); print(f"pushbtn up1: rob y={o[1]:.4f} M=({o[20]:.4f},{o[21]:.4f}) C=({o[9]:.4f},{o[10]:.4f})")
o=probe(s,(1,0),1.0); print(f"then +x: rob=({o[0]:.4f},{o[1]:.4f}) M=({o[20]:.4f},{o[21]:.4f})")
o=probe(s,(0,1),1.0); print(f"then up: rob y={o[1]:.4f} M=({o[20]:.4f},{o[21]:.4f})")
s.close()

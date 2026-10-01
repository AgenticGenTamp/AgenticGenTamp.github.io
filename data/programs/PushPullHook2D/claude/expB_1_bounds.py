import numpy as np, itertools
from helper import Sim

def probe(s, d, step=0.005, maxn=600):
    d=np.array(d,float)
    for i in range(maxn):
        p=s.obs[:2].copy()
        s.step([d[0]*step,d[1]*step,0,0,0.0])
        if np.linalg.norm(s.obs[:2]-p)<1e-9:
            # refine with 0.001
            for j in range(10):
                p=s.obs[:2].copy(); s.step([d[0]*0.001,d[1]*0.001,0,0,0.0])
                if np.linalg.norm(s.obs[:2]-p)<1e-9: break
            return s.obs[:2].copy()
    return s.obs[:2].copy()

DIRS={'+x':(1,0),'-x':(-1,0),'+y':(0,1),'-y':(0,-1)}
START=(2.2,0.7)
for arm in [0.1,0.2]:
    for dn,d in DIRS.items():
        base=np.arctan2(d[1],d[0])
        for lbl,th in [('fwd',base),('back',base+np.pi),('perp',base+np.pi/2)]:
            s=Sim(42)
            r=s.goto(START[0],START[1],th,arm,0.0)
            if r<0: print("stuck setup",arm,dn,lbl); s.close(); continue
            th_act=s.obs[2]
            p=probe(s,d)
            key = p[0] if dn[1]=='x' else p[1]
            print(f"arm={arm} dir={dn} {lbl:4s} th={th_act:+.3f} limit={key:.4f}  pos=({p[0]:.4f},{p[1]:.4f}) steps={s.n}")
            s.close()

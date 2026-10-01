import sys; sys.path.insert(0,'.')
import numpy as np, math, time
from approach import *
from kin import ik_down, ik_refine, Q0
rng=np.random.default_rng(0)
gain_steps=[]; T=0; n=0
for trial in range(200):
    x=rng.uniform(-0.2,0.2); y=rng.uniform(-0.5,0.5); yaw=rng.uniform(-3,3)
    base=(-0.6,float(np.clip(y*0.5,-0.2,0.2)),0.0)
    qc=Q0+rng.uniform(-0.5,0.5,7)
    q,ok=ik_down(base,np.array([x,y,0.793]),qc,yaw=yaw)
    if not ok: continue
    d0=np.abs(wrap(q-qc)).max()
    t=time.time(); q2=ik_refine(base,np.array([x,y,0.793]),q,qc,yaw=yaw); T+=time.time()-t
    d1=np.abs(wrap(q2-qc)).max(); n+=1
    gain_steps.append(math.ceil(d0/0.2-1e-9)-math.ceil(d1/0.2-1e-9))
print(n, np.mean(gain_steps), np.bincount(np.array(gain_steps)-min(gain_steps)), min(gain_steps), T/n*1000)

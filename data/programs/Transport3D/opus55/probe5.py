from env_client import make_env
import numpy as np
from kin import fk_arm, ik
env = make_env()
J=['joint_%d'%i for i in range(1,8)]
def q(obs):
    r=obs.get_object_from_name('robot'); return np.array([float(obs.get(r,f)) for f in J])
obs,_=env.reset(seed=1)
Rdown=np.array([[0,1,0],[1,0,0],[0,0,-1.]])  # from home orientation approx
def goto(qt, maxstep=0.1):
    global obs
    for _ in range(200):
        qc=q(obs); d=qt-qc
        if np.max(np.abs(d))<1e-3: return True
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(d,-maxstep,maxstep)
        obs,*_=env.step(a)
        if np.max(np.abs(q(obs)-qc))<1e-6: return False
    return False
qc=q(obs)
for z in np.arange(0.3,-0.6,-0.02):
    qt,ep,er=ik(np.array([0.5,0,z]),Rdown,qc)
    ok=goto(qt,0.02)
    print(round(z,3), ok, round(ep,4), np.round(fk_arm(q(obs))[:3,3],3))
    if not ok: break
    qc=q(obs)
env.close()

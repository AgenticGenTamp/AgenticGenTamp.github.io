import numpy as np, sys
from env_client import make_env
from kin import *
np.set_printoptions(precision=4, suppress=True, linewidth=150)
env = make_env()
obs, info = env.reset(seed=0)
base = obs[125:128].copy(); q = obs[128:135].copy()
print("base", base, "EE world", fk_world(base, q)[:3,3])
# target: empty spot on island: world (0.95, 0.15)
def goto(qd, n=60, grip=0.0):
    global obs
    for t in range(n):
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(qd-obs[128:135],-0.1,0.1); a[10]=grip
        obs,*_=env.step(a)
        if np.max(np.abs(obs[128:135]-qd))<2e-3: break
    return obs[128:135]
xy=np.array([0.95,0.15])
qc=q.copy()
for z in np.arange(0.60,0.35,-0.01):
    qd,err=ik(base,qc,np.array([xy[0],xy[1],z]),R_down(base[2]))
    qa=goto(qd,n=40)
    pa=fk_world(base,qa)[:3,3]
    print(f"z={z:.3f} ikerr={err:.4f} jerr={np.max(np.abs(qa-qd)):.4f} actual_tool={pa}")
    qc=qd
env.close()

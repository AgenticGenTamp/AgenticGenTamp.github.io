import numpy as np
from env_client import make_env
from ik import ik
np.set_printoptions(precision=4,suppress=True,linewidth=200)
Rdown=np.array([[1,0,0],[0,-1,0],[0,0,-1]],dtype=float)

def run(xy, zs, seed=0):
    env=make_env(); obs,_=env.reset(seed=seed)
    q=obs[128:135].copy()
    log=[]
    for z in zs:
        qd,err=ik(np.array([xy[0],xy[1],z]),Rdown,q)
        for i in range(25):
            cur=obs[128:135]
            a=np.zeros(11)
            a[3:10]=np.clip((qd-cur)*1.0,-0.1,0.1)
            obs,r,t,tr,_=env.step(a)
            if np.max(np.abs(qd-obs[128:135]))<0.01: break
        cur=obs[128:135]
        # world objects
        cubes=obs[:80].reshape(5,16)[:,:3]
        wip=obs[147:150]
        log.append((z, np.max(np.abs(qd-cur)), i, wip.copy(), cubes.copy(), obs[125:128].copy()))
    env.close()
    return log

zs=np.arange(0.35,-0.45,-0.05)
log=run((0.343,0.317),zs)
w0=log[0][3]; c0=log[0][4]
for z,qe,it,w,c,b in log:
    print(f"z={z:+.2f} qerr={qe:.3f} it={it} dwiper={np.round(w-w0,3)} maxdcube={np.abs(c-c0).max():.3f} base={np.round(b,3)}")

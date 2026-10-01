import numpy as np
from ik import ik
RDOWN=np.array([[1,0,0],[0,-1,0],[0,0,-1]],dtype=float)
JLO=np.array([-1e9,-2.18,-1e9,-2.54,-1e9,-2.07,-1e9])
JHI=np.array([ 1e9, 1.24, 1e9, 2.54, 1e9, 2.07, 1e9])

def move(env, obs, p, R=RDOWN, steps=200, grip=None, tol=0.005, base=None, qd=None):
    if qd is None:
        qd,_=ik(np.array(p),R,obs[128:135].copy())
    qd=np.clip(qd,JLO,JHI)
    g = obs[135] if grip is None else grip
    used=0; hist=[]
    for i in range(steps):
        a=np.zeros(11); a[3:10]=np.clip(qd-obs[128:135],-0.1,0.1); a[10]=g
        if base is not None: a[0:3]=base
        obs,r,t,tr,_=env.step(a); used+=1
        err=np.abs(qd-obs[128:135]).max(); hist.append(err)
        if err<tol: break
        if len(hist)>12 and hist[-12]-err < 0.004: break
    return obs, np.abs(qd-obs[128:135]).max(), used

RFWD=np.array([[0,0,1],[0,1,0],[-1,0,0]],dtype=float)  # tool z -> local +x, tool x -> local -z

"""Descend the EE at several base-frame xy and find the stall height (contact)."""
from env_client import make_env
import numpy as np, kin, sys
env = make_env()
obs, info = env.reset(seed=3, options={'object_count':10})
R = obs.get_object_from_name('robot')
base = obs.data[R][:3].copy()
print('base', np.round(base,3))
TOOL=0.12
q = obs.data[R][3:10].copy()
def goto(tgt, n=60, tol=0.008):
    global obs,q
    qd,err = kin.ik(np.array(tgt), kin.rot_down(), q, TOOL)
    for k in range(n):
        qc = obs.data[R][3:10]
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(qd-qc,-0.1,0.1); a[10]=0.0
        obs,_,_,_,_=env.step(a)
        if np.abs(qd-obs.data[R][3:10]).max()<tol: break
    q = obs.data[R][3:10].copy()
    return err, np.abs(qd-q).max(), k
for xy in [(0.35,0.0),(0.5,0.0),(0.5,-0.25)]:
    goto((xy[0],xy[1],0.40))
    for z in np.arange(0.35,0.05,-0.02):
        err,qerr,k = goto((xy[0],xy[1],z),40)
        stat = 'STALL' if qerr>0.03 else 'ok'
        print('xy',xy,'z',round(z,2),'ikerr',round(err,3),'qerr',round(qerr,3),stat)
        if qerr>0.05: break
    goto((xy[0],xy[1],0.40))
env.close()

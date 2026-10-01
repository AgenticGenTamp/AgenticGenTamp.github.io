"""Sweep EE laterally just above the counter and detect when the scoop moves."""
from env_client import make_env
import numpy as np, kin, sys
Z = float(sys.argv[1]); FX = float(sys.argv[2]); TOOL=float(sys.argv[3])
env = make_env()
obs, info = env.reset(seed=3, options={'object_count':10})
R = obs.get_object_from_name('robot'); SC = obs.get_object_from_name('scoop_0')
base = obs.data[R][:3].copy(); th=base[2]
scoop0 = obs.data[SC][:3].copy()
dx,dy = scoop0[0]-base[0], scoop0[1]-base[1]
fx = np.cos(th)*dx+np.sin(th)*dy; fy=-np.sin(th)*dx+np.cos(th)*dy
print('base',np.round(base,3),'scoop base-frame', round(fx,3), round(fy,3))
q = obs.data[R][3:10].copy()
def goto(qd, n=40):
    global obs,q
    for k in range(n):
        qc = obs.data[R][3:10]
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(qd-qc,-0.1,0.1); a[10]=0.0
        obs,_,_,_,_=env.step(a)
        if np.abs(qd-obs.data[R][3:10]).max()<0.008: break
    q = obs.data[R][3:10].copy()
    return k
# start above, then descend
for (tx,ty,tz) in [(FX,-0.30,Z+0.15),(FX,-0.30,Z)]:
    qd,err = kin.ik(np.array([tx,ty,tz]), kin.rot_down(), q, TOOL)
    print('goto',tx,ty,tz,'ikerr',round(err,4),'steps',goto(qd,60),'qerr',round(np.abs(qd-q).max(),3))
for ty in np.arange(-0.28, 0.32, 0.04):
    qd,err = kin.ik(np.array([FX,ty,Z]), kin.rot_down(), q, TOOL)
    st=goto(qd,25)
    d = obs.data[SC][:3]-scoop0
    print('ty',round(ty,2),'ikerr',round(err,3),'qerr',round(np.abs(qd-q).max(),3),'scoop delta',np.round(d,3))
env.close()

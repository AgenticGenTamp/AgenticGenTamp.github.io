"""Locate the gripper by contacting the scoop: hold EE at scoop xy in an assumed
arm frame and sweep z downward, watching for scoop motion."""
from env_client import make_env
import numpy as np, kin, sys

Z_ARM = float(sys.argv[1]) if len(sys.argv)>1 else 0.40
env = make_env()
obs, info = env.reset(seed=3, options={'object_count':10})
R = obs.get_object_from_name('robot'); SC = obs.get_object_from_name('scoop_0')
def rob(o): return o.data[R]
def sc(o): return o.data[SC][:3].copy()
scoop0 = sc(obs)
base = rob(obs)[:3]
print('base', np.round(base,3), 'scoop', np.round(scoop0,3))
# assume arm base at base xy, height Z_ARM, yaw = base yaw
th = base[2]
dx, dy = scoop0[0]-base[0], scoop0[1]-base[1]
fx = np.cos(th)*dx + np.sin(th)*dy
fy = -np.sin(th)*dx + np.cos(th)*dy
print('scoop in base frame', round(fx,3), round(fy,3))
TOOL=0.12
q = rob(obs)[3:10].copy()
for ztgt in np.arange(0.45, -0.05, -0.02):
    tgt = np.array([fx, fy, ztgt])
    qd, err = kin.ik(tgt, kin.rot_down(), q, TOOL)
    if err > 0.02:
        print('unreachable', round(ztgt,2), round(err,3)); continue
    for k in range(40):
        qcur = rob(obs)[3:10]
        dq = np.clip(qd - qcur, -0.1, 0.1)
        a = np.zeros(11, dtype=np.float32); a[3:10]=dq; a[10]=0.0
        obs,rew,term,trunc,info = env.step(a)
        if np.abs(qd-rob(obs)[3:10]).max() < 0.01: break
    q = rob(obs)[3:10].copy()
    d = np.linalg.norm(sc(obs)-scoop0)
    print('z',round(ztgt,3),'steps',k,'qerr',round(np.abs(qd-q).max(),3),'scoop moved',round(d,4), np.round(sc(obs),3))
    if d > 0.01:
        print('CONTACT at z(arm frame)=',round(ztgt,3), 'world scoop z', scoop0[2]); break
env.close()

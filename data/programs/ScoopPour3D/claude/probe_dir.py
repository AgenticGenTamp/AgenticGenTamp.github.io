"""Approach the scoop from one direction at low height with gripper closed; report first-contact EE cmd pos."""
from env_client import make_env
import numpy as np, kin, sys
DIR = sys.argv[1]; Z=float(sys.argv[2]); GRIP=float(sys.argv[3])
env = make_env()
obs, info = env.reset(seed=3, options={'object_count':10})
R = obs.get_object_from_name('robot'); SC = obs.get_object_from_name('scoop_0')
base = obs.data[R][:3].copy(); th=base[2]
scoop0 = obs.data[SC][:3].copy()
dx,dy = scoop0[0]-base[0], scoop0[1]-base[1]
fx = np.cos(th)*dx+np.sin(th)*dy; fy=-np.sin(th)*dx+np.cos(th)*dy
TOOL=0.12
q = obs.data[R][3:10].copy()
def goto(tgt, n=60, tol=0.008):
    global obs,q
    qd,err = kin.ik(np.array(tgt), kin.rot_down(), q, TOOL)
    for k in range(n):
        qc = obs.data[R][3:10]
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(qd-qc,-0.1,0.1); a[10]=GRIP
        obs,_,_,_,_=env.step(a)
        if np.abs(qd-obs.data[R][3:10]).max()<tol: break
    q = obs.data[R][3:10].copy()
    return err, np.abs(qd-q).max()
S=0.30
if DIR=='-y': start=(fx,fy-S); step=(0,0.015)
if DIR=='+y': start=(fx,fy+S); step=(0,-0.015)
if DIR=='-x': start=(fx-S,fy); step=(0.015,0)
if DIR=='+x': start=(fx+S,fy); step=(-0.015,0)
goto((start[0],start[1],0.32))
goto((start[0],start[1],Z))
p=np.array(start,dtype=float)
for i in range(40):
    p = p + np.array(step)
    e,qe = goto((p[0],p[1],Z),20)
    d = obs.data[SC][:3]-scoop0
    if np.linalg.norm(d)>0.004:
        print(DIR,'CONTACT cmd',np.round(p,3),'scoopdelta',np.round(d,3),'qerr',round(qe,3)); break
else:
    print(DIR,'no contact', np.round(p,3))
print(DIR,'scoop base-frame center', round(fx,3), round(fy,3), 'scoop quat', np.round(obs.data[SC][3:7],3))
env.close()

"""Calibrate z0 and xy offsets using stalls over counter / bin interior / bin rim."""
from env_client import make_env
import numpy as np, kin
env = make_env()
obs, info = env.reset(seed=3, options={'object_count':10})
R = obs.get_object_from_name('robot'); YB=obs.get_object_from_name('bin_yellow_0')
CUBES=[o for o in obs.data if o.name.startswith('cube_')]
def base(): return obs.data[R][:3].copy()
def qq(): return obs.data[R][3:10].copy()
TOOL=0.12; G=[1.0]; QD=[None]
def step(bc=(0,0,0)):
    global obs
    a=np.zeros(11,dtype=np.float32); a[:3]=np.clip(bc,-0.1,0.1); a[10]=G[0]
    if QD[0] is not None: a[3:10]=np.clip(QD[0]-qq(),-0.1,0.1)
    obs,rew,term,trunc,info=env.step(a)
def arm_to(tgt,yaw=0.0,n=60,tol=0.008):
    qd,err=kin.ik(np.array(tgt), kin.rot_down(yaw), qq(), TOOL); QD[0]=qd
    for k in range(n):
        step()
        if np.abs(qd-qq()).max()<tol: break
    return err, np.abs(qd-qq()).max()
def base_to(bx,by,bth=0.0,n=40,tol=0.01):
    for k in range(n):
        b=base(); e=np.array([bx-b[0],by-b[1],bth-b[2]])
        if np.abs(e).max()<tol: break
        step(np.clip(e,-0.1,0.1))
base_to(-0.16, -0.20, 0.0)
b=base(); print('base',np.round(b,3))
cz = np.mean([obs.data[c][2] for c in CUBES]); print('cube z', round(cz,4))
def descend(wx, wy, z0=0.30, label=''):
    b=base(); fx=wx-b[0]; fy=wy-b[1]
    arm_to((fx,fy,z0))
    prev=z0
    for z in np.arange(z0,0.10,-0.01):
        e,qe = arm_to((fx,fy,z),30)
        if qe>0.03:
            print(label,'stall at z_arm',round(z,3),'qerr',round(qe,3)); arm_to((fx,fy,z0)); return z
    print(label,'no stall'); arm_to((fx,fy,z0)); return None
# over bare counter far from bins (x=0.5,y=0.0 is the gap between bins)
descend(0.30, -0.20, label='counter@x0.30')
descend(0.50, 0.00, label='gap between bins')
descend(0.50, -0.20, label='yellow bin interior')
env.close()

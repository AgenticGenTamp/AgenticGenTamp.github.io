import numpy as np
from calib_util import R, RD
import kin
def yaw(q): w,x,y,z=q; return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
def rz(t): return kin.rotz(t)
def lin(r, pw, Rw=RD, vmax=0.006, stop=None, maxsteps=300, settle=3):
    """Cartesian straight-line move of tool point using IK each step. stop(r)->bool aborts."""
    p0,_=kin.fk_world(r.base(), r.qi)
    pw=np.array(pw,float); d=pw-p0; n=max(1,int(np.ceil(np.linalg.norm(d)/vmax)))
    for i in range(1,n+1):
        q,e=r.ik_world(p0+d*i/n, Rw)
        dq=(q-r.qi)/0.25
        r.step(dq=np.clip(dq,-0.1,0.1))
        if stop is not None and stop(r): return False
    for k in range(maxsteps):
        q,e=r.ik_world(pw,Rw); dq=(q-r.qi)/0.25
        if np.max(np.abs(dq))<1e-5: break
        r.step(dq=np.clip(dq,-0.1,0.1))
        if stop is not None and stop(r): return False
    for k in range(settle): r.step(dq=np.zeros(7))
    return True
def local2world(r, u, v, name='scoop_0'):
    P=r.P(name); th=yaw(r.Q(name)); c,s=np.cos(th),np.sin(th)
    return P[:2]+np.array([c*u-s*v,s*u+c*v]), th

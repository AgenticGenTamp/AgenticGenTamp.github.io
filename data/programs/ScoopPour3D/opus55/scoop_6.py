import numpy as np, sys
from scoop_util import *
import calib_util
A=[]
_orig=calib_util.R.step
def step(s, db=(0,0,0), dq=None):
    a=np.zeros(11); a[0:3]=np.clip(db,-.1,.1)
    if dq is not None: a[3:10]=np.clip(dq,-.1,.1)
    a[10]=s.g; A.append(a); return _orig(s,db,dq)
calib_util.R.step=step
seed=0; u,v,dy,zt=0,0,0,0.49
r=R(seed); r.gripper(0.0,5)
pw,th=local2world(r,u,v); Rw=rz(th+dy)@RD
lin(r,[pw[0],pw[1],0.53],Rw,vmax=0.01); lin(r,[pw[0],pw[1],zt],Rw,vmax=0.004)
r.gripper(1.0,20); r.qi=r.q().copy()
lin(r,[pw[0],pw[1],0.65],Rw,vmax=0.006)
# present: rotate about world z so scoop long axis faces camera variants, and tilt
n0=len(A)
Rv=rz(0)@RD
lin(r,[0.30,0.0,0.68],Rv,vmax=0.006); print('pose1',len(A),r.P('scoop_0').round(3),r.Q('scoop_0').round(3))
for k in range(8): r.step(dq=np.zeros(7))
Rv2=rz(np.pi/2)@RD
lin(r,[0.30,0.0,0.68],Rv2,vmax=0.006); print('pose2',len(A),r.P('scoop_0').round(3),r.Q('scoop_0').round(3))
for k in range(8): r.step(dq=np.zeros(7))
Rv3=rz(np.pi/2)@RD@kin.rotz(0)  # tilt about tool x
Rt=rz(np.pi/2)@RD@np.array([[1,0,0],[0,np.cos(0.9),-np.sin(0.9)],[0,np.sin(0.9),np.cos(0.9)]])
lin(r,[0.30,0.0,0.68],Rt,vmax=0.006); print('pose3',len(A),r.P('scoop_0').round(3),r.Q('scoop_0').round(3))
for k in range(8): r.step(dq=np.zeros(7))
np.save('scoop_render/actions.npy',np.array(A)); print(len(A))

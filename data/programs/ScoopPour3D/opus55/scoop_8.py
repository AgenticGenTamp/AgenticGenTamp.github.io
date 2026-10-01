import numpy as np, sys
from scoop_util import *
seed=int(sys.argv[1]); ddx=float(sys.argv[2]); ddy=float(sys.argv[3]); zoff=float(sys.argv[4]) if len(sys.argv)>4 else 0.0
tag=' '.join(sys.argv[1:])
r=R(seed); r.gripper(0.0,5)
cubes=[n for n in r.obs.get_object_names() if n.startswith('cube')]
C=np.array([r.P(n) for n in cubes]); ctr=C[:,:2].mean(0)
pw,th=local2world(r,0,0); Rg=rz(th)@RD
lin(r,[pw[0],pw[1],0.53],Rg,vmax=0.01); lin(r,[pw[0],pw[1],0.49],Rg,vmax=0.004)
r.gripper(1.0,20); r.qi=r.q().copy()
lin(r,[pw[0],pw[1],0.60],Rg,vmax=0.006)
b=r.base(); tgt=np.array([-0.15,ctr[1],0.0])
for k in range(60):
    d=tgt-r.base()
    if np.abs(d).max()<0.003: break
    r.step(db=d,dq=np.zeros(7))
S0=r.P('scoop_0')
start=np.array([ctr[0]-ddx*0.5,ctr[1]-ddy*0.5])
ok=lin(r,[start[0],start[1],0.60],RD,vmax=0.008)
print(tag,'held? scoop z %.3f'%r.P('scoop_0')[2],'yaw %.2f'%yaw(r.Q('scoop_0')),flush=True)
zt=0.4775+0.034+0.002+zoff
lin(r,[start[0],start[1],zt],RD,vmax=0.004,maxsteps=20)
r.qi=r.q().copy()
lin(r,[start[0]+ddx,start[1]+ddy,zt],RD,vmax=0.003,maxsteps=20)
r.qi=r.q().copy()
lin(r,[start[0]+ddx,start[1]+ddy,0.60],RD,vmax=0.004)
for k in range(15): r.step(dq=np.zeros(7))
P=r.P('scoop_0'); Rs=kin.quat2mat(r.Q('scoop_0'))
C=np.array([r.P(n) for n in cubes]); up=C[:,2]>0.5
print(tag,'scoop',P.round(3),'tilt',Rs[2].round(2),'cubes up',up.sum())
for c in C[up]: print('   local',(Rs.T@(c-P)).round(3))

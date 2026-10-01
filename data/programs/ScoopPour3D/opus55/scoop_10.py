import numpy as np, sys
from scoop_util import *
seed=int(sys.argv[1]); tilt=np.radians(float(sys.argv[2])); drag=float(sys.argv[3]); ug=float(sys.argv[4]) if len(sys.argv)>4 else -0.035
tag=' '.join(sys.argv[1:])
def Ry(t): c,s=np.cos(t),np.sin(t); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
r=R(seed); r.gripper(0.0,5)
cubes=[n for n in r.obs.get_object_names() if n.startswith('cube')]
C=np.array([r.P(n) for n in cubes]); ctr=C[:,:2].mean(0)
pw,th=local2world(r,ug,0); Rg=rz(th)@RD
lin(r,[pw[0],pw[1],0.53],Rg,vmax=0.01); lin(r,[pw[0],pw[1],0.48],Rg,vmax=0.004)
r.gripper(1.0,20); r.qi=r.q().copy()
lin(r,[pw[0],pw[1],0.62],Rg,vmax=0.006)
tgt=np.array([-0.15,ctr[1],0.0])
for k in range(60):
    d=tgt-r.base()
    if np.abs(d).max()<0.003: break
    r.step(db=d,dq=np.zeros(7))
R0=rz(np.pi)@RD
x0=ctr[0]+drag/2+0.03
lin(r,[x0,ctr[1],0.62],R0,vmax=0.008)
Rt=Ry(-tilt)@R0
lin(r,[x0,ctr[1],0.62],Rt,vmax=0.008)
tp,TR=r.fk(); P=r.P('scoop_0'); Rs=kin.quat2mat(r.Q('scoop_0'))
print(tag,'held z %.3f'%P[2],'rel',(TR.T@(P-tp)).round(3),'scoop z-axis',Rs[:,2].round(2),'u-axis',Rs[:,0].round(2),flush=True)
# descend until contact
def stop(r): return np.abs(r.q()-r.qi).max()>0.012
lin(r,[x0,ctr[1],0.47],Rt,vmax=0.002,stop=stop,maxsteps=10)
zc=r.fk()[0][2]; print(tag,'contact tool z %.3f scoop z %.3f'%(zc,r.P('scoop_0')[2]),flush=True)
r.qi=r.q().copy()
zd=zc+0.004
lin(r,[x0,ctr[1],zd],Rt,vmax=0.003)
lin(r,[x0-drag,ctr[1],zd],Rt,vmax=0.003,maxsteps=10)
r.qi=r.q().copy()
# pitch back upright while lifting
n=25
for i in range(1,n+1):
    s=i/n; q,e=r.ik_world([x0-drag-0.01*s,ctr[1],zd+0.03*s],Ry(-tilt*(1-s))@R0); r.step(dq=np.clip((q-r.qi)/0.25,-.1,.1))
lin(r,[x0-drag,ctr[1],0.62],R0,vmax=0.004)
for k in range(15): r.step(dq=np.zeros(7))
P=r.P('scoop_0'); Rs=kin.quat2mat(r.Q('scoop_0'))
C=np.array([r.P(n) for n in cubes]); up=C[:,2]>0.5
print(tag,'RESULT cubes up',up.sum(),'scoop',P.round(3),'zaxis',Rs[:,2].round(2))
for c in C[up]: print('   local',(Rs.T@(c-P)).round(3))

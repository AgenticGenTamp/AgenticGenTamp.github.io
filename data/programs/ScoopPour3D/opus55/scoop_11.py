import numpy as np, sys
from scoop_util import *
seed=int(sys.argv[1]); tilt=np.radians(float(sys.argv[2])); drag=float(sys.argv[3]); sgn=float(sys.argv[4]) if len(sys.argv)>4 else 1.0
ug=-0.035
tag=' '.join(sys.argv[1:])
def Rx(t): c,s=np.cos(t),np.sin(t); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
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
R0=RD
# drag direction: -y*sgn ; roll so opening faces drag dir
y0=ctr[1]+sgn*(drag/2+0.02); xt=ctr[0]-0.065
Rt=Rx(sgn*tilt)@R0
lin(r,[xt,y0,0.62],Rt,vmax=0.008)
for k in range(10): r.step(dq=np.zeros(7))
tp,TR=r.fk(); P=r.P('scoop_0'); Rs=kin.quat2mat(r.Q('scoop_0'))
print(tag,'held rel',(TR.T@(P-tp)).round(3),'scoop z-axis',Rs[:,2].round(2),'u-axis',Rs[:,0].round(2),flush=True)
lin(r,[xt,y0,0.56],Rt,vmax=0.004)
for k in range(8): r.step(dq=np.zeros(7))
z=r.fk()[0][2]; zs=[]; tp0,TR0=r.fk(); rel0=TR0.T@(r.P('scoop_0')-tp0); lag=0
for k in range(150):
    z-=0.0012
    q,e=r.ik_world([xt,y0,z],Rt); r.step(dq=np.clip((q-r.qi)/0.25,-.1,.1))
    tp,TR=r.fk(); zs.append(tp[2]); rel=TR.T@(r.P('scoop_0')-tp)
    if k>6 and (zs[-5]-zs[-1]<0.002 or np.linalg.norm(rel-rel0)>0.003): break
zc=r.fk()[0][2]; print(tag,'contact tool z %.3f scoop z %.3f'%(zc,r.P('scoop_0')[2]),'k',k,'lag',np.abs(r.q()-r.qi).max().round(3),'relchg',np.linalg.norm(rel-rel0).round(4),flush=True)
r.qi=r.q().copy(); zd=zc+0.004
lin(r,[xt,y0,zd],Rt,vmax=0.003)
lin(r,[xt,y0-sgn*drag,zd],Rt,vmax=0.003,maxsteps=10)
P=r.P('scoop_0'); Rs=kin.quat2mat(r.Q('scoop_0')); print(tag,'after drag zaxis',Rs[:,2].round(2),'origin',P.round(3))
for n_ in cubes:
    c=r.P(n_)
    if np.linalg.norm(c-P)<0.09: print('   near local',(Rs.T@(c-P)).round(3))
r.qi=r.q().copy()
n=25; yf=y0-sgn*drag
for i in range(1,n+1):
    s=i/n; q,e=r.ik_world([xt,yf-sgn*0.01*s,zd+0.03*s],Rx(sgn*tilt*(1-s))@R0); r.step(dq=np.clip((q-r.qi)/0.25,-.1,.1))
lin(r,[xt,yf,0.62],R0,vmax=0.004)
for k in range(15): r.step(dq=np.zeros(7))
P=r.P('scoop_0'); Rs=kin.quat2mat(r.Q('scoop_0'))
C=np.array([r.P(n) for n in cubes]); up=C[:,2]>0.5
print(tag,'RESULT cubes up',up.sum(),'scoop',P.round(3),'zaxis',Rs[:,2].round(2))
for c in C[up]: print('   local',(Rs.T@(c-P)).round(3))

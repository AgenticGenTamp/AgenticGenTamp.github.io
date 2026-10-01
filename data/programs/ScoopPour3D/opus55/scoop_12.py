import numpy as np, sys
from scoop_util import *
seed=int(sys.argv[1]); beta=np.radians(float(sys.argv[2])); gam=np.radians(float(sys.argv[3])); drag=float(sys.argv[4])
ug=float(sys.argv[5]) if len(sys.argv)>5 else -0.035; zg=float(sys.argv[6]) if len(sys.argv)>6 else 0.482
tag=' '.join(sys.argv[1:])
r=R(seed); r.gripper(0.0,5)
cubes=[n for n in r.obs.get_object_names() if n.startswith('cube')]
C=np.array([r.P(n) for n in cubes]); ctr=C[:,:2].mean(0)
S0=kin.quat2mat(r.Q('scoop_0')); us,vs,zs=S0[:,0],S0[:,1],S0[:,2]
tz=np.cos(beta)*zs-np.sin(beta)*us; tx=vs; ty=np.cross(tz,tx); T0=np.stack([tx,ty,tz],1)
pw,th=local2world(r,ug,0); g=np.array([pw[0],pw[1],zg])
lin(r,g+0.07*tz,T0,vmax=0.01); lin(r,g,T0,vmax=0.004)
r.gripper(1.0,20); r.qi=r.q().copy()
lin(r,g+np.array([0,0,0.12]),T0,vmax=0.006)
tp,TR=r.fk(); P=r.P('scoop_0'); Sg=kin.quat2mat(r.Q('scoop_0'))
lifted=P[2]>0.52; print(tag,'lifted',lifted,'scoop z %.3f'%P[2],'tilt chg',(S0.T@Sg).round(2)[2],flush=True)
if not lifted: sys.exit()
Rrel=Sg.T@TR; prel=TR.T@(P-tp)   # tool orient in scoop frame; scoop origin in tool frame
def S(g_):
    z=np.array([-np.cos(g_),0,np.sin(g_)]); v=np.array([0,-1.,0]); u=np.cross(v,z); return np.stack([u,v,z],1)
tgt=np.array([-0.15,ctr[1],0.0])
for k in range(60):
    d=tgt-r.base()
    if np.abs(d).max()<0.003: break
    r.step(db=d,dq=np.zeros(7))
def toolpose(origin,g_):
    Tw=S(g_)@Rrel; return origin-Tw@prel, Tw
x0=ctr[0]+drag/2; 
p,Tw=toolpose(np.array([x0,ctr[1],0.62]),gam); lin(r,p,Tw,vmax=0.008)
for k in range(8): r.step(dq=np.zeros(7))
tp,TR=r.fk(); P=r.P('scoop_0'); Sc=kin.quat2mat(r.Q('scoop_0')); print(tag,'posed: scoop z-axis',Sc[:,2].round(2),'u',Sc[:,0].round(2),'err',np.linalg.norm(P-np.array([x0,ctr[1],0.62])).round(3),flush=True)
prelc=TR.T@(P-tp)
zs_=[]; z=tp[2]
for k in range(200):
    z-=0.0012
    q,e=r.ik_world([tp[0],tp[1],z],Tw); r.step(dq=np.clip((q-r.qi)/0.25,-.1,.1))
    t2,_=r.fk(); zs_.append(t2[2])
    if k>6 and zs_[-5]-zs_[-1]<0.002: break
t2,TR=r.fk(); print(tag,'contact: scoop origin z %.3f tool z %.3f'%(r.P('scoop_0')[2],t2[2]),flush=True)
r.qi=r.q().copy(); zd=t2[2]+0.003
lin(r,[tp[0],tp[1],zd],Tw,vmax=0.003)
lin(r,[tp[0]-drag,tp[1],zd],Tw,vmax=0.003,maxsteps=10)
P=r.P('scoop_0'); Rs=kin.quat2mat(r.Q('scoop_0')); print(tag,'after drag origin',P.round(3),'zaxis',Rs[:,2].round(2))
for n_ in cubes:
    c=r.P(n_)
    if np.linalg.norm(c-P)<0.08: print('   near local',(Rs.T@(c-P)).round(3))
r.qi=r.q().copy()
t2,_=r.fk(); n=25
for i in range(1,n+1):
    s=i/n; g_=gam+(np.radians(50)-gam)*s
    p,Tw=toolpose(r.P('scoop_0')*0+np.array([P[0],P[1],P[2]+0.04*s]),g_)
    q,e=r.ik_world(p,Tw); r.step(dq=np.clip((q-r.qi)/0.25,-.1,.1))
p,Tw=toolpose(np.array([P[0],P[1],0.62]),np.radians(50)); lin(r,p,Tw,vmax=0.004)
for k in range(15): r.step(dq=np.zeros(7))
P=r.P('scoop_0'); Rs=kin.quat2mat(r.Q('scoop_0'))
C=np.array([r.P(n) for n in cubes]); up=C[:,2]>0.5
print(tag,'RESULT cubes up',up.sum(),'scoop',P.round(3),'zaxis',Rs[:,2].round(2))
for c in C[up]: print('   local',(Rs.T@(c-P)).round(3))

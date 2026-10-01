from env_client import make_env
from kinematics import fk,ik
from scipy.spatial.transform import Rotation
import numpy as np
E=make_env();s,_=E.reset(seed=0);steps=0
RDOWN=np.diag([1.,-1.,-1.])
def rob():
 o=s.get_object_from_name('robot');return np.array([s.get(o,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint'+str(i) for i in range(1,8)]]),np.array([s.get(o,'vel_arm_joint'+str(i)) for i in range(1,8)])
def pose(n):
 o=s.get_object_from_name(n);p=np.array([s.get(o,f) for f in ['x','y','z']]);q=np.array([s.get(o,f) for f in ['qx','qy','qz','qw']]);return p,Rotation.from_quat(q).as_matrix()
def report(label):
 r,_=rob();T=fk(r[3:],r[:3]);p,R=pose('scoop_0');c=np.array([pose(n)[0] for n in s.get_object_names() if n.startswith('cube_')]);print(label,'step',steps,'ee',np.round(T[:3,3],4),'scoop',np.round(p,4),'quat',np.round(Rotation.from_matrix(R).as_quat(),3),'relative',np.round(T[:3,:3].T@(p-T[:3,3]),4),'cubes',np.round(c.min(0),3),np.round(c.max(0),3),'green',int(sum(c[:,1]>0)),flush=True)
def go(label,p,grip=1,R=RDOWN,limit=90):
 global s,steps
 r,_=rob();qt,err=ik(p,R,r[3:],r[:3]);print('target',label,p,'err',err,flush=True)
 for k in range(limit):
  r,v=rob();d=qt-r[3:];a=np.zeros(11);a[3:10]=np.clip(2.5*d+.15*v,-.1,.1);a[-1]=grip;s,rw,t,tr,i=E.step(a);steps+=1
  if k>8 and max(abs(d))<.006:break
 report(label)
p,_=pose('scoop_0');report('initial')
go('above',[p[0],p[1],.6],0)
go('down',[p[0],p[1],.48],0)
go('close',[p[0],p[1],.48],1,limit=15)
go('lift',[p[0],p[1],.65],1)
r,_=rob();T=fk(r[3:],r[:3]);sp,sR=pose('scoop_0');rel=T[:3,:3].T@(sp-T[:3,3]);print('grasp rel',rel,flush=True)
def scoopgo(label,sp,R=RDOWN,limit=90):go(label,np.array(sp)-R@rel,1,R,limit)
scoopgo('over_front',[.65,-.2,.65])
scoopgo('lower',[.65,-.2,.485])
scoopgo('sweep1',[.55,-.2,.485])
scoopgo('sweep2',[.45,-.2,.485])
scoopgo('raise',[.45,-.2,.65])
scoopgo('green',[.5,.2,.65])
scoopgo('dump',[.5,.2,.65],Rotation.from_euler('y',-1.5).as_matrix()@RDOWN)
E.close()

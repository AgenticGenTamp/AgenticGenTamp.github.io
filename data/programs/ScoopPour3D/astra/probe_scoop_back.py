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
r,_=rob(); bx=r[0]-.1
for k in range(40):
 r,_=rob();a=np.zeros(11);a[0]=np.clip(2.5*(bx-r[0]),-.1,.1);s,rw,t,tr,i=E.step(a);steps+=1
 if abs(r[0]-bx)<.001:break
p,sR=pose('scoop_0');report('initial')
yaw=Rotation.from_matrix(sR).as_euler('xyz')[2]
RDOWN=Rotation.from_euler('z',yaw).as_matrix()@np.diag([1.,-1.,-1.])
handle=p+sR@np.array([0.,0,0])
go('above',[handle[0],handle[1],.6],0,RDOWN)
go('down',[handle[0],handle[1],.48],0,RDOWN)
go('close',[handle[0],handle[1],.48],1,RDOWN,limit=15)
go('lift',[handle[0],handle[1],.65],1,RDOWN)
r,_=rob();T=fk(r[3:],r[:3]);sp,sR=pose('scoop_0');rel=T[:3,:3].T@(sp-T[:3,3]);print('grasp rel',rel,flush=True)
if sp[2]<.55: E.close();raise SystemExit('Grasp failed')
def scoopgo(label,target,Robj,limit=70):
 global s,steps
 target=np.array(target);qt=None
 for k in range(limit):
  r,v=rob();T=fk(r[3:],r[:3]);sp,sR=pose('scoop_0')
  if k%12==0:
   rel=T[:3,:3].T@(sp-T[:3,3]);relR=T[:3,:3].T@sR
   Rtool=Robj@relR.T;qt,err=ik(target-Rtool@rel,Rtool,r[3:],r[:3])
   if k==0:print('scoop_target',label,target,'ikerr',err,flush=True)
  d=qt-r[3:];act=np.zeros(11);act[3:10]=np.clip(2.5*d+.15*v,-.1,.1);act[-1]=1
  s,rw,t,tr,i=E.step(act);steps+=1
  if k>10 and np.linalg.norm(sp-target)<.008 and np.linalg.norm(Rotation.from_matrix(Robj@sR.T).as_rotvec())<.10:break
 report(label)
RBACK=Rotation.from_euler('z',np.pi).as_matrix()
scoopgo('level_back',[sp[0],sp[1],.68],RBACK,100)
scoopgo('clearance',[.30,-.2,.73],RBACK)
scoopgo('over_back',[.38,-.2,.65],RBACK)
scoopgo('lower_back',[.38,-.2,.49],RBACK)
scoopgo('sweep1',[.46,-.2,.49],RBACK)
scoopgo('sweep2',[.55,-.2,.49],RBACK)
scoopgo('raise',[.49,-.2,.69],RBACK)
scoopgo('green',[.49,.2,.69],RBACK)
scoopgo('dump',[.49,.2,.65],Rotation.from_euler('y',1.4).as_matrix()@RBACK)
E.close()

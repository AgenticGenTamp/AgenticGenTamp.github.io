from env_client import make_env
import numpy as np
from scipy.spatial.transform import Rotation
import math

e=make_env();o,_=e.reset(seed=1);r=o.get_object_from_name('robot');b=o.get_object_from_name('green1'); xy=np.array([o.get(b,'pose_x'),o.get(b,'pose_y')]);fs=['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]
def move(target,close=0):
 global o
 for _ in range(80):
  cur=np.array([o.get(r,f) for f in fs]);d=np.array(target)-cur
  for i in [2,7,9]:d[i]=(d[i]+math.pi)%(2*math.pi)-math.pi
  if max(abs(d))<1e-4:return
  a=np.zeros(11);a[:10]=np.clip(d,-.2,.2);a[10]=close;o,*_=e.step(a)
  if max(abs(np.array([o.get(r,f) for f in fs])-cur))<1e-5:return
q=[0,.9,0,-1.2,math.pi,-.3,math.pi]
pose=[xy[0]+1.5,xy[1]+.188,math.pi]+q;move(pose,1)
pose[0]=xy[0]+.825;move(pose,1)
a=np.zeros(11);a[10]=-1;o,*_=e.step(a)
print('grasp',o.get(r,'grasp_active'))
pose[0]+=1.;move(pose)
def measured():
 p=np.array([o.get(b,'pose_'+ax) for ax in ['x','y','z']]);rb=Rotation.from_quat([o.get(b,'pose_'+ax) for ax in ['qx','qy','qz','qw']]).as_matrix();t=np.array([o.get(r,'grasp_tf_'+ax) for ax in ['x','y','z']]);rt=Rotation.from_quat([o.get(r,'grasp_tf_'+ax) for ax in ['qx','qy','qz','qw']]).as_matrix();R=rb@rt.T;p=p-R@t
 return p,R

def fk(q,variant):
 x=-.05 if variant else 0.;tool=.18 if variant else .13
 p=np.array([pose[0],pose[1],0.]);R=Rotation.from_euler('z',pose[2]).as_matrix()
 for tran,axis,angle in [([x,.188,.990675],'z',q[0]),([.1,0,0],'y',q[1]),([0,0,0],'x',q[2]),([.4,0,0],'y',q[3]),([0,0,0],'x',q[4]),([.321,0,0],'y',q[5]),([0,0,0],'x',q[6]),([tool,0,0],'x',0.)]:
  p=p+R@tran;R=R@Rotation.from_euler(axis,angle).as_matrix()
 return p,R
for i in [0,1,2,3,4,5,6]:
 pose[3+i]+=.2;move(pose);actualq=np.array([o.get(r,'joint_'+str(j)) for j in range(1,8)]);p,R=measured()
 print('change',i,'q',actualq.tolist(),'tool',p.tolist(),'errs',[(float(np.linalg.norm(p-fk(actualq,v)[0])),float(np.linalg.norm(R-fk(actualq,v)[1]))) for v in [0,1]],flush=True)
e.close()

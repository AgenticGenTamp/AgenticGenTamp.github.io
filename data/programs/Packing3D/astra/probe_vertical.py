from env_client import make_env
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import brentq
params=[([0,0,.15643],np.pi),([0,.005375,-.12838],np.pi/2),([0,-.21038,-.006375],-np.pi/2),([0,.006375,-.21038],np.pi/2),([0,-.20843,-.006375],-np.pi/2),([0,0,-.10593],np.pi/2),([0,-.10593,0],-np.pi/2)]
def fk(q):
 t=np.eye(4)
 for (xyz,rx),qi in zip(params,q):
  a=np.eye(4);a[:3,3]=xyz;a[:3,:3]=R.from_euler('x',rx).as_matrix()@R.from_euler('z',qi).as_matrix();t=t@a
 return t[:3,3]+t[:3,:3]@np.array([.000261641841,0,-.181524928])+[.119899991,.0000852158346,-.00520012073]
def qs(q2): return [0,q2,-np.pi,-2.05185294,0,q2+2.05185294-np.pi,np.pi/2]
E=make_env()
for seed,name,opt in [(0,'part0',None),(1,'part0',None),(0,'part0',{'object_count':1})]:
 found=False
 for z in [.145,.135,.125,.155,.165,.115]:
  q2=brentq(lambda q2:fk(qs(q2))[2]-z,-.5,1.0);q=qs(q2);ep=fk(q)
  s,info=E.reset(seed=seed,options=opt);p=s.get_object_from_name(name);robot=s.get_object_from_name('robot');px=s.get(p,'pose_x');py=s.get(p,'pose_y')
  off=.025 if p.type.name=='Kinematic3DTriangle' and s.get(p,'triangle_type')>.5 else -.005
  dest=np.r_[px+off-ep[0],py+off-ep[1],0,q]
  for i in range(8):
   cur=np.array([s.get(robot,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+[f'joint_{j}' for j in range(1,8)]])
   a=np.r_[np.clip(dest-cur,-.2,.2),-1];s,*_=E.step(a)
   if s.get(p,'grasp_active'):
    print('SUCCESS',seed,opt,'z',z,'q2',q2,'target',dest.tolist(),'robot',s.data[robot].tolist(),flush=True);found=True;break
  if found:break
  print('FAIL',seed,opt,z,'q2',q2,flush=True)
E.close()

from env_client import make_env
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
params=[([0,0,.15643],np.pi),([0,.005375,-.12838],np.pi/2),([0,-.21038,-.006375],-np.pi/2),([0,.006375,-.21038],np.pi/2),([0,-.20843,-.006375],-np.pi/2),([0,0,-.10593],np.pi/2),([0,-.10593,0],-np.pi/2)]
def fk(q):
 t=np.eye(4)
 for (xyz,rx),qi in zip(params,q):
  a=np.eye(4);a[:3,3]=xyz;a[:3,:3]=R.from_euler('x',rx).as_matrix()@R.from_euler('z',qi).as_matrix();t=t@a
 return t[:3,3]+t[:3,2]*(-.18)
e=make_env();s,_=e.reset(seed=0);rob=s.get_object_from_name('robot');p=s.get_object_from_name('part0');q0=np.array([s.get(rob,'joint_'+str(i)) for i in range(1,8)])
def move(target):
 global s
 for _ in range(50):
  cur=np.array([s.get(rob,'pos_base_x'),s.get(rob,'pos_base_y'),s.get(rob,'pos_base_rot')]+[s.get(rob,'joint_'+str(i)) for i in range(1,8)])
  delta=target-cur
  if np.max(np.abs(delta))<1e-5:break
  a=np.r_[np.clip(delta,-.2,.2),-1.]
  ns,*_=e.step(a)
  if all(abs(ns.get(rob,f)-s.get(rob,f))<1e-6 for f in ['pos_base_x','pos_base_y']+['joint_'+str(i) for i in range(1,8)]):break
  s=ns
  if s.get(rob,'grasp_active'):
   print('GRASP',[(f,s.get(rob,f)) for f in e.observation_space.type_features[rob.type]],flush=True);return True
 return False
for length in [.18,.12,.06,.24]:
 for z in [.095,.125,.155,.195,.225]:
  target=np.array([s.get(p,'pose_x')+.12,0,z])
  def fun(v):
   q=q0.copy();q[[1,3,5]]=v
   return np.r_[fk(q)-target, .015*(v-q0[[1,3,5]])]
  sol=least_squares(fun,q0[[1,3,5]],max_nfev=80)
  q=q0.copy();q[[1,3,5]]=sol.x
  print('TRY',length,z,q.round(3),fk(q).round(3),flush=True)
  for dx in [0,-.05,.05,-.1,.1]:
   if move(np.r_[-.12+dx,s.get(p,'pose_y')-.001,0,q]):e.close();raise SystemExit
print('NONE');e.close()

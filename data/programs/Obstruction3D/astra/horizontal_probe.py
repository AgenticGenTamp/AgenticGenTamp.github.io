import numpy as np
from scipy.optimize import least_squares
from env_client import make_env
from kinematics import fk
home=np.array([0,-.35,-np.pi,-2.5,0,-.87,np.pi/2])
orient=np.array([[0,0,-1],[1,0,0],[0,-1,0]])
features=['pos_base_x','pos_base_y','pos_base_rot']+['joint_'+str(i) for i in range(1,8)]
for length in [.20,.12,.061525]:
 e=make_env();s,_=e.reset(seed=0);r=s.get_object_from_name('robot');b=s.get_object_from_name('target_block')
 p=np.array([s.get(b,'pose_'+f) for f in 'xyz'])
 def vec():return np.array([s.get(r,f) for f in features])
 def step(a):
  global s
  s,*_=e.step(np.asarray(a,dtype=np.float32))
  if s.get(r,'grasp_active'):
   print('SUCCESS',length,[(f,s.get(r,f)) for f in e.observation_space.type_features[r.type]],flush=True);e.close();raise SystemExit
 def move(goal):
  for i in range(90):
   old=vec();delta=goal-old;a=np.zeros(11);a[:10]=delta*min(1,.1/max(1e-9,max(abs(delta))));a[10]=1;step(a)
   if max(abs(vec()-old))<1e-7 or max(abs(goal-vec()))<1e-4:break
  return max(abs(goal-vec()))
 # Walk around the near end of the tabletop before facing backwards.
 for bx,by,theta in [(-.12,-.6,0),(.8,-.6,0),(.8,-.6,np.pi),(.8,p[1],np.pi)]:
  goal=np.r_[bx,by,theta,home];err=move(goal)
  print('BASE',length,np.round(vec()[:3],3),round(err,3),flush=True)
 base=vec()[:3];c=np.cos(base[2]);ss=np.sin(base[2]);R=np.array([[c,-ss,0],[ss,c,0],[0,0,1]])
 local=R.T@(p-np.r_[base[:2],0])
 q=home.copy()
 for z in [.3,.2,.15,.12,.1]:
  goalp=local.copy();goalp[2]=z
  def residual(q):
   T=fk(q);return np.r_[T[:3,3]+T[:3,:3]@[0,0,-length]-goalp,(T[:3,:3]-orient).ravel()*.25]
  sol=least_squares(residual,q,max_nfev=150);goal=np.r_[base,sol.x];err=move(goal);q=vec()[3:]
  print('HEIGHT',length,z,'err',round(err,3),'TCP',np.round(fk(q)[:3,3]+fk(q)[:3,:3]@[0,0,-length],3),flush=True)
  a=np.zeros(11);a[-1]=-1;step(a)
 # Small base offsets to correct unknown tool calibration.
 for dx,dy in [(0,-.03),(0,.03),(-.03,0),(.03,0),(-.03,-.03),(.03,-.03)]:
  goal=vec();goal[:2]=base[:2]+[dx,dy];move(goal);a=np.zeros(11);a[-1]=-1;step(a)
 e.close()

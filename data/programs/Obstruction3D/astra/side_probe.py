import numpy as np
from scipy.optimize import least_squares
from kinematics import fk
from env_client import make_env
E=make_env();s,_=E.reset(seed=0);r=s.get_object_from_name('robot');b=s.get_object_from_name('target_block')
qhome=np.array([0,-.35,-np.pi,-2.5,0,-.87,np.pi/2])
def rv():return np.array([s.get(r,'pos_base_x'),s.get(r,'pos_base_y'),s.get(r,'pos_base_rot')]+[s.get(r,'joint_'+str(i+1)) for i in range(7)])
def move(goal):
 global s
 for i in range(100):
  cur=rv();d=goal-cur;a=np.r_[d*min(1,.08/max(abs(d))),0]
  ns,*_=E.step(a);s=ns
  if max(abs(rv()-goal))<1e-4:return True
  if max(abs(rv()-cur))<1e-6:return False
 return False
def solve(x,z,ang,L):
 def f(u):
  q=qhome.copy();q[[1,3,5]]=u;T=fk(q)
  return np.r_[T[:3,3]+T[:3,:3]@[0,0,-L]-[x,.001,z],T[0,2]-np.sin(ang),T[2,2]-np.cos(ang)]
 a=least_squares(f,[.5,-1.5,-.9],max_nfev=100)
 q=qhome.copy();q[[1,3,5]]=a.x
 return q
for ang in [-np.pi/2,np.pi/2,-np.pi/4,np.pi/4]:
 for L in [.061525,.10,.14,.18,.22]:
  s,_=E.reset(seed=0);p=np.array([s.get(b,'pose_'+v) for v in 'xyz'])
  goal=np.r_[p[0]-.5,p[1]-.001,0,solve(.5,.4,ang,L)]
  ok=move(goal)
  for z in np.arange(.4,.06,-.01):
   goal[3:]=solve(.5,z,ang,L);ok=move(goal)
   a=np.zeros(11);a[-1]=-1;s,*_=E.step(a)
   if s.get(r,'grasp_active'):
    print('GRASP',ang,L,z,rv(),{f:s.get(r,f) for f in E.observation_space.type_features[r.type]},flush=True);E.close();raise SystemExit
   if not ok: print('BLOCKED',ang,L,z,rv(),flush=True);break
E.close()

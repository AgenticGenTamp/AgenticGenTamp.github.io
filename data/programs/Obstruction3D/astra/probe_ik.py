import numpy as np
from scipy.optimize import least_squares
from kinematics import fk
from env_client import make_env
qhome=np.array([0,-.35,-np.pi,-2.5,0,-.87,np.pi/2])
def solve(x,z,L):
 def fun(u):
  q=qhome.copy();q[[1,3,5]]=u;T=fk(q)
  return np.r_[T[:3,3]+T[:3,:3]@[0,0,-L]-[x,.001,z],T[0,2]]
 r=least_squares(fun, qhome[[1,3,5]],max_nfev=100)
 q=qhome.copy();q[[1,3,5]]=r.x
 return q
E=make_env();s,_=E.reset(seed=0)
r=s.get_object_from_name('robot');b=s.get_object_from_name('target_block')
def rv(s):return np.array([s.get(r,'pos_base_x'),s.get(r,'pos_base_y'),s.get(r,'pos_base_rot')]+[s.get(r,'joint_'+str(i+1)) for i in range(7)])
def move(goal,grip=0):
 global s
 for i in range(60):
  cur=rv(s);a=np.r_[np.clip(goal-cur,-.199,.199),grip]
  ns,_,term,trunc,inf=E.step(a);diff=np.max(np.abs(rv(ns)-cur));s=ns
  if term: print('SOLVED',flush=True)
  if np.max(np.abs(rv(s)-goal))<.001:return True
  if diff<1e-5:return False
 return False
for L in [.12,.16,.20,.24,.28]:
 for z in [.10,.13,.07]:
  s,_=E.reset(seed=0)
  # First lift and reach while above objects, then lower.
  bpos=np.array([s.get(b,'pose_'+v) for v in 'xyz'])
  goal=np.r_[bpos[0]-.50,bpos[1]-.001,0,solve(.5,.35,L)]
  ok=move(goal)
  goal[3:]=solve(.5,z,L)
  ok2=move(goal,-1)
  print('trial',L,z,ok,ok2,'q',rv(s)[[4,6,8]],'grasp',s.get(r,'grasp_active'),flush=True)
  if s.get(r,'grasp_active'):
   for typ in E.observation_space.types:
    for o in s.get_objects(typ): print(o.name,{f:s.get(o,f) for f in E.observation_space.type_features[typ]},flush=True)
   E.close();raise SystemExit
E.close()

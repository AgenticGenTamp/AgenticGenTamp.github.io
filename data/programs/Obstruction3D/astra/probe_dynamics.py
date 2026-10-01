import numpy as np
from scipy.optimize import least_squares
from env_client import make_env
from kinematics import fk
qhome=np.array([0,-.35,-np.pi,-2.5,0,-.87,np.pi/2]);off=np.array([.119899783,.000348925,-.005199842]);L=.181525034
for name in ['obstruction0','obstruction1']:
 e=make_env();s,_=e.reset(seed=0);r=s.get_object_from_name('robot');b=s.get_object_from_name(name)
 p=np.array([s.get(b,'pose_'+f) for f in 'xyz']);base=np.r_[p[0]-.50,p[1]-.001-off[1],0]
 def rv():return np.array([s.get(r,'pos_base_x'),s.get(r,'pos_base_y'),s.get(r,'pos_base_rot')]+[s.get(r,'joint_'+str(i+1)) for i in range(7)])
 def move(goal):
  global s
  for k in range(100):
   cur=rv();d=goal-cur
   if max(abs(d))<1e-5:return True
   a=np.r_[d*min(1,.08/max(abs(d))),0];s,*_=e.step(a)
   if max(abs(rv()-cur))<1e-6:return False
  return False
 def solve(z):
  def fun(u):
   q=qhome.copy();q[[1,3,5]]=u;T=fk(q)
   return np.r_[T[:3,3]+T[:3,:3]@[0,0,-L]+off+np.r_[base[:2],0]-[p[0],p[1],z],T[0,2],T[2,2]-1]
  sol=least_squares(fun,[.5,-1.5,-.9]);q=qhome.copy();q[[1,3,5]]=sol.x;return q
 goal=np.r_[base,solve(.40)];print('HOVER',name,move(goal),flush=True)
 for dz in [.065,.060,.055,.050,.045,.040,.035]:
  goal[3:]=solve(p[2]+dz);ok=move(goal)
  for close in [1,-1]:a=np.zeros(11);a[-1]=close;s,*_=e.step(a)
  print('TRY',name,dz,ok,'GRASP',s.get(r,'grasp_active'),'Q',rv(),'TF',[s.get(r,'grasp_tf_'+f) for f in 'xyz'],flush=True)
  if s.get(r,'grasp_active'):break
 e.close()

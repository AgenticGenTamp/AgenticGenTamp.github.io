import numpy as np
from scipy.optimize import least_squares
from kinematics import fk
from env_client import make_env
E=make_env();s,_=E.reset(seed=0,options={"object_count":0});r=s.get_object_from_name('robot');b=s.get_object_from_name('target_block');region=s.get_object_from_name('target_region')
qhome=np.array([0,-.35,-np.pi,-2.5,0,-.87,np.pi/2])
def pos(o):return np.array([s.get(o,'pose_'+v) for v in 'xyz'])
def rv():return np.array([s.get(r,'pos_base_x'),s.get(r,'pos_base_y'),s.get(r,'pos_base_rot')]+[s.get(r,'joint_'+str(i+1)) for i in range(7)])
def move(goal,grip=-1):
 global s
 for i in range(100):
  cur=rv();d=goal-cur
  if max(abs(d))<1e-5:return True
  a=np.r_[d*min(1,.05/max(abs(d))),grip];s,*res=E.step(a)
  if res[1]: print('TERMINATED',res,flush=True)
  if max(abs(rv()-cur))<1e-6:return False
 return False
def solve(z):
 def f(u):
  q=qhome.copy();q[[1,3,5]]=u;T=fk(q)
  return np.r_[T[:3,3]+T[:3,:3]@[0,0,-.061525]-[.5,.001,z],T[0,2]-np.sin(np.pi/4),T[2,2]-np.cos(np.pi/4)]
 a=least_squares(f,[.5,-1.5,-.9],max_nfev=100);q=qhome.copy();q[[1,3,5]]=a.x;return q
p=pos(b);goal=np.r_[p[0]-.5,p[1]-.001,0,solve(.4)];move(goal,0)
for z in np.arange(.4,.22,-.01):
 goal[3:]=solve(z);ok=move(goal);s,*_=E.step(np.r_[np.zeros(10),-1])
 if s.get(r,'grasp_active'):print('GRASP',z,pos(b),flush=True);graspz=z;break
else:print('FAILED GRASP',flush=True);E.close();raise SystemExit
for z in [graspz+.10]:
 goal[3:]=solve(z);ok=move(goal);print('LIFT',z,ok,pos(b),flush=True)
dest=pos(region);goal[:2]+=dest[:2]-pos(b)[:2];ok=move(goal);print('TRANSPORT',ok,pos(b),flush=True)
# Lower onto region while keeping rigid orientation.
for z in list(np.arange(graspz+.10,graspz+.014,-.005))+[graspz+.01001]:
 goal[3:]=solve(z);ok=move(goal);print("LOWER",round(z,3),ok,pos(b),flush=True)
 if not ok:break
s,rew,t,tr,info=E.step(np.r_[np.zeros(10),1]);print('RELEASE',pos(b),'term',t,flush=True)
for i in range(3):s,rew,t,tr,info=E.step(np.zeros(11));print('WAIT',i,pos(b),'term',t,flush=True)
E.close()

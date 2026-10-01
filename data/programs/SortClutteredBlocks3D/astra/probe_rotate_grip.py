from env_client import make_env
from kinematics import fk
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import numpy as np

def xyz(s,o):return np.array([s.get(o,f) for f in ['x','y','z']])
cache={}
def ik(z):
 if z not in cache:
  def err(q):
   p,r=fk(q)
   return np.r_[(p+r@np.array([0,0,.12])-np.array([.55,0,z]))*3,Rotation.from_matrix(r.T@np.diag([1,-1,-1])).as_rotvec()]
  cache[z]=least_squares(err,np.array([-.04,.7,3.22,-1.4,-.1,-1.1,.08]),max_nfev=100).x
 return cache[z]+np.array([0,0,0,0,0,0,np.pi/2])
for z in [.00,.02,.04,.06]:
 e=make_env();s,_=e.reset(seed=0);r=s.get_object_from_name('robot');o=s.get_object_from_name('cube1')
 def move(base,z,grip,n):
  global s
  for _ in range(n):
   q=np.array([s.get(r,'pos_arm_joint'+str(j)) for j in range(1,8)]);v=np.array([s.get(r,'vel_arm_joint'+str(j)) for j in range(1,8)])
   b=np.array([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']]);a=np.r_[(base-b)/[.87,.87,.994],2.5*(ik(z)-q)+.15*v,grip];s,*_=e.step(np.clip(a,e.action_space.low,e.action_space.high))
 for dx,dy in [(0,0),(.025,0),(-.025,0),(0,.025),(0,-.025)]:
  center=xyz(s,o);base=np.r_[center[:2]+[.55+dx,dy],np.pi]
  move(base,.2,0,85 if dx==dy==0 else 35);move(base,z,0,40);before=xyz(s,o)
  move(base,z,1,6);closed=xyz(s,o);move(base,.2,1,40);lift=xyz(s,o)
  print('z',z,'offset',dx,dy,'close',np.round(closed-before,4),'lift',np.round(lift-before,4),'qerr',np.round(np.array([s.get(r,'pos_arm_joint'+str(j)) for j in range(1,8)])-ik(.2),3),flush=True)
  if lift[2]>.47:
   e.close();raise SystemExit('LIFTED')
  move(base,.2,0,3)
 e.close()

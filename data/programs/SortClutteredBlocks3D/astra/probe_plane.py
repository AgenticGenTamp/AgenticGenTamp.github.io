from env_client import make_env
import numpy as np
from kinematics import fk
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
E=make_env();s,_=E.reset(seed=0);rb=s.get_object_from_name('robot')
qhome=np.deg2rad([0,-20,180,-146,0,-50,90])
def move(q,base,steps=100,grip=0):
 global s
 for i in range(steps):
  cur=np.array([s.get(rb,'pos_arm_joint'+str(j)) for j in range(1,8)]);v=np.array([s.get(rb,'vel_arm_joint'+str(j)) for j in range(1,8)])
  a=np.r_[(base-np.array([s.get(rb,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']]))/np.array([.87,.87,.994]),2.5*(q-cur)+.15*v,grip]
  s,r,d,tr,inf=E.step(np.clip(a,E.action_space.low,E.action_space.high))
 return cur
for z in [.2,.0,-.2,-.3]:
 def err(q):
  p,r=fk(q);return np.r_[(p+r@np.array([0,0,.12])-np.array([.55,0,z]))*3,Rotation.from_matrix(r.T@np.diag([1,-1,-1])).as_rotvec()]
 q=least_squares(err,qhome,max_nfev=200).x;qhome=q
 cur=move(q,np.array([.55,0,np.pi]))
 print('HEIGHT',z,'q',q.round(3),'err',(cur-q).round(3),'predicted',fk(cur)[0]+fk(cur)[1]@np.array([0,0,.12]),flush=True)
 print('MOVABLE',[(n,[round(s.get(s.get_object_from_name(n),f),3) for f in ['x','y','z']]) for n in sorted(s.get_object_names()) if n.startswith(('cube','bin'))],flush=True)
E.close()

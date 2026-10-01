from env_client import make_env
from kinematics import fk
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import numpy as np
import time
QFEATURE=['pos_arm_joint'+str(i) for i in range(1,8)]
VFEATURE=['vel_arm_joint'+str(i) for i in range(1,8)]
def qget(s,r):return np.array([s.get(r,f) for f in QFEATURE])
def xyz(s,o):return np.array([s.get(o,f) for f in ['x','y','z']])
def solve(q0,p):
 goal=np.diag([1.,-1.,-1.])
 def loss(q):
  fp,fr=fk(q);return np.r_[fp+fr@np.array([0,0,.12])-p,.3*Rotation.from_matrix(goal.T@fr).as_rotvec()]
 sol=least_squares(loss,q0,max_nfev=100)
 return sol.x,np.linalg.norm(loss(sol.x))
def servo(s,r,targetq,targetbase,g):
 a=np.zeros(11); a[:3]=np.array(targetbase)-np.array([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']]);a[:2]/=.8699
 a[3:10]=2.5*(targetq-qget(s,r))+.15*np.array([s.get(r,f) for f in VFEATURE]);a=np.clip(a,-.1,.1);a[-1]=g;return a
for h in [.38,.39,.40,.41,.42]:
 e=make_env();s,_=e.reset(seed=0);r=s.get_object_from_name('robot');o=s.get_object_from_name('cube1');p0=xyz(s,o);base=np.r_[p0[:2]+[.55,0],np.pi]
 seed=np.array([-.058,.672,3.251,-1.315,-.074,-1.159,.057]); highq,_=solve(seed,np.array([.55,0,.2])); tq,err=solve(highq,np.array([.55,0,p0[2]-h]));start=time.time()
 for t in range(100):s,_,_,_,_=e.step(servo(s,r,highq,base,0))
 for t in range(120):s,_,_,_,_=e.step(servo(s,r,tq,base,0))
 before=xyz(s,o);actualq=qget(s,r)
 for t in range(5):s,_,_,_,_=e.step(servo(s,r,tq,base,1))
 closed=xyz(s,o);base[0]+=.12
 for t in range(8):s,_,_,_,_=e.step(servo(s,r,tq,base,1))
 print('h',h,'ikerr',round(err,4),'q',np.round(tq,3),'qerr',np.round(actualq-tq,3),'pre',np.round(before-p0,3),'closed',np.round(closed-before,3),'drag',np.round(xyz(s,o)-closed,3),'sec',round(time.time()-start,2),flush=True)
 e.close()

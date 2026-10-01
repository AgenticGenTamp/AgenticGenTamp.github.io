import numpy as np
from scipy.optimize import least_squares
from kinematics import HOME,fk
from env_client import make_env
from concurrent.futures import ThreadPoolExecutor

def pose(x,z):
 def res(v):
  q=HOME.copy();q[[1,3,5]]=v;p,R=fk(q)
  return np.r_[(p-np.array([x,.001,z]))[[0,2]],R[0,2]]
 q=HOME.copy();q[[1,3,5]]=least_squares(res,HOME[[1,3,5]],max_nfev=100).x
 return q

def run(par):
 x,z=par;e=make_env();s,inf=e.reset(seed=42);r=s.get_object_from_name('robot');o=s.get_object_from_name('cube1');b0=np.array([s.get(o,'x')-.625,s.get(o,'y')-.001,0])
 def move(q,b,g,n):
  nonlocal s
  for i in range(n):
   cur=np.array([s.get(r,'pos_arm_joint'+str(j)) for j in range(1,8)]);base=np.array([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
   a=np.r_[np.clip(b-base,-.1,.1),np.clip(2*(q-cur),-.1,.1),g].astype(np.float32)
   s,rew,t,tr,inf=e.step(a)
   if t: print('SUCCESS',par,'reward',rew,flush=True)
  return np.round([s.get(o,f) for f in ['x','y','z']],4)
 move(HOME,b0,0,35);down=pose(.5,-.04);move(down,b0,0,95);move(down,b0,1,15)
 print(par,'lift',move(pose(x,z-.055),b0,1,100),flush=True)
 dest=np.array([1.5-x-.125,-.001,0.])
 print(par,'insert',move(pose(x,z-.055),dest,1,80),flush=True)
 print(par,'drop',move(pose(x,z-.055),dest,0,30),flush=True);e.close()
if __name__=='__main__':
 with ThreadPoolExecutor(3) as ex:list(ex.map(run,[(.7,z) for z in [.22,.35,.5,.65,.8]]))

import numpy as np
from scipy.optimize import least_squares
from concurrent.futures import ThreadPoolExecutor
from env_client import make_env
from kinematics import fk, HOME

def pose(z,reach_x=.5):
 def residual(x):
  q=HOME.copy();q[[1,3,5]]=x;p,R=fk(q)
  return np.r_[(p-[reach_x,.001,z])[[0,2]],R[0,2]]
 sol=least_squares(residual,HOME[[1,3,5]],max_nfev=100)
 q=HOME.copy();q[[1,3,5]]=sol.x;return q

def trial(reach):
 z=.5
 e=make_env();s,_=e.reset(seed=42)
 r=s.get_objects(e.observation_space.get_type('mujoco_tidybot_robot'))[0]
 obs=s.get_objects(e.observation_space.get_type('mujoco_movable_object'))
 o=min(obs,key=lambda o:s.get(o,'y'));xyz=np.array([s.get(o,f) for f in ['x','y','z']])
 pickup=np.r_[xyz[:2]-[.625,.001],0.];place=np.array([1.5-reach-.125,-.001,0.])
 low=pose(-.04);high=pose(z-.055,reach);out=[]
 for step in range(340):
  target=HOME if step<35 else (low if step<140 else high)
  base=pickup if step<240 else place.copy()
  if step>=310:base[0]-=.25
  q=np.array([s.get(r,'pos_arm_joint'+str(i)) for i in range(1,8)])
  b=np.array([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
  a=np.r_[np.clip(base-b,-.1,.1),np.clip((target-q)*2,-.1,.1),float(125<=step<295)]
  s,re,te,tr,info=e.step(a.astype(np.float32))
  if step in [139,239,294,309,339]:out.append((step,np.round([s.get(o,f) for f in ['x','y','z']],4).tolist(),re,te,np.round(b,3).tolist()))
 e.close();print('reach',reach,out,flush=True)
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(trial,[.65,.75]))

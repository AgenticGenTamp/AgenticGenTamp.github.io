import numpy as np
from scipy.optimize import least_squares
from concurrent.futures import ThreadPoolExecutor
from env_client import make_env
from kinematics import fk, HOME

def trial(xoffset):
 z=-.04
 e=make_env();s,_=e.reset(seed=42)
 r=s.get_objects(e.observation_space.get_type('mujoco_tidybot_robot'))[0]
 obs=s.get_objects(e.observation_space.get_type('mujoco_movable_object'))
 o=min(obs,key=lambda o:s.get(o,'y'))
 xyz=np.array([s.get(o,f) for f in ['x','y','z']])
 base=np.r_[xyz[:2]-[xoffset,.001],0.]
 def residual(x):
  q=HOME.copy();q[[1,3,5]]=x;p,R=fk(q)
  return np.r_[(p-[.5,.001,z])[[0,2]],R[0,2]]
 sol=least_squares(residual,HOME[[1,3,5]],max_nfev=100)
 lower=HOME.copy();lower[[1,3,5]]=sol.x
 outputs=[]
 for step in range(220):
  target=HOME if step<35 or step>=140 else lower
  q=np.array([s.get(r,'pos_arm_joint'+str(i)) for i in range(1,8)])
  b=np.array([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
  a=np.r_[np.clip(base-b,-.1,.1),np.clip((target-q)*2,-.1,.1),float(step>=125)]
  s,re,te,tr,info=e.step(a.astype(np.float32))
  if step in [34,124,139,219]:outputs.append((step,[round(s.get(o,f),4) for f in ['x','y','z']],np.round(fk(q)[0],3).tolist()))
 e.close();print('offset',xoffset,'res',sol.cost,'q',lower.tolist(),'trajectory',outputs,flush=True)
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(trial,[.60,.625,.65,.675]))

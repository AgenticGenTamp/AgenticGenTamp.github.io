import numpy as np
from scipy.optimize import least_squares
from concurrent.futures import ThreadPoolExecutor
from env_client import make_env
from kinematics import fk, HOME

def trial(z):
 e=make_env();s,_=e.reset(seed=42)
 r=s.get_objects(e.observation_space.get_type('mujoco_tidybot_robot'))[0]
 obs=s.get_objects(e.observation_space.get_type('mujoco_movable_object'))
 o=min(obs,key=lambda o:s.get(o,'y'));xyz=np.array([s.get(o,f) for f in ['x','y','z']])
 def residual(x):
  q=HOME.copy();q[[1,3,5]]=x;p,R=fk(q)
  return np.r_[(p-[.5,.001,z])[[0,2]],R[0,2]]
 sol=least_squares(residual,HOME[[1,3,5]],max_nfev=100)
 lower=HOME.copy();lower[[1,3,5]]=sol.x
 points=[np.r_[xyz[:2]+[x,y],0] for iy,y in enumerate(np.arange(-.45,.451,.09)) for x in (np.arange(-.75,.751,.075) if iy%2==0 else np.arange(.75,-.751,-.075))]
 outputs=[];mx=0
 for step in range(100+len(points)*2+70):
  target=lower if step<100+len(points)*2 else HOME
  ix=max(0,min(len(points)-1,(step-100)//2));base=points[ix]
  q=np.array([s.get(r,'pos_arm_joint'+str(i)) for i in range(1,8)])
  b=np.array([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
  a=np.r_[np.clip(base-b,-.1,.1),np.clip((target-q)*2,-.1,.1),float(step%4>=1 or step>=100+len(points)*2)]
  s,re,te,tr,info=e.step(a.astype(np.float32));pos=np.array([s.get(o,f) for f in ['x','y','z']]);d=np.linalg.norm(pos-xyz)
  if d>mx+.01:mx=d;outputs.append((step,np.round(b,3).tolist(),np.round(pos,3).tolist()))
 e.close();print('z',z,'motion',outputs,'final',pos,flush=True)
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(trial,[-.04,.25]))

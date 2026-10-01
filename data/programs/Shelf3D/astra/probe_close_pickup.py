from probe_tilt import pose
from kinematics import HOME
from env_client import make_env
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def run(x):
 e=make_env();s,inf=e.reset(seed=42);r=s.get_object_from_name('robot');o=s.get_object_from_name('cube1');cup=s.get_object_from_name('cupboard_1');b=np.array([s.get(o,'x')-x-.125,s.get(o,'y')-.001,0.])
 def move(q,g,n):
  nonlocal s
  for i in range(n):
   cur=np.array([s.get(r,'pos_arm_joint'+str(j)) for j in range(1,8)]);base=np.array([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
   s,rew,t,tr,inf=e.step(np.r_[np.clip(b-base,-.1,.1),np.clip(2*(q-cur),-.1,.1),g])
  return np.round([s.get(o,f) for f in ['x','y','z']],4).tolist(),round(float(np.max(abs(cur-q))),4)
 d=pose(x,-.04);u=pose(.65,s.get(cup,'z')+.61,-np.pi/2)
 move(HOME,0,35);print(x,'lower',move(d,0,95),flush=True);print(x,'grasp',move(d,1,10),flush=True);print(x,'lift',move(u,1,65),flush=True)
 e.close()
with ThreadPoolExecutor(3) as ex:list(ex.map(run,[.35,.4,.45]))

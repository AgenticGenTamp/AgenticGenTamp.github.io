from env_client import make_env
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def trial(q4):
 e=make_env();s,i=e.reset(seed=1);ro=s.get_object_from_name('robot');cu=s.get_object_from_name('cube_0');fs=['pos_arm_joint'+str(j) for j in range(1,8)]
 goal=np.array([0,2.4,np.pi,q4,0,-.873,np.pi/2]);orig=np.array([s.get(cu,f) for f in ['x','y','z']]);old=orig.copy()
 for k in range(225):
  q=np.array([s.get(ro,f) for f in fs]);a=np.zeros(11,dtype=np.float32);a[3:10]=np.clip((goal-q)*2,-.1,.1);a[10]=1;a[2]=np.clip((-np.pi/2-s.get(ro,'pos_base_rot'))*1.1,-.1,.1)
  if k>85:a[0]=np.clip((orig[0]-s.get(ro,'pos_base_x'))*1.1,-.1,.1)
  if k>=140:a[1]=-.025
  s,r,t,tr,i=e.step(a);cur=np.array([s.get(cu,f) for f in ['x','y','z']])
  if np.linalg.norm(cur-old)>.003 or k in [139,224]:
   print(q4,k,'BASE',np.round([s.get(ro,'pos_base_x'),s.get(ro,'pos_base_y')],3).tolist(),'Q',np.round(q,3).tolist(),'CUBE',np.round(cur,4).tolist(),'R',r,'TERM',t,flush=True);old=cur.copy()
  if t:break
 e.close()
with ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(trial,[-.5,-.3,0,.3]))

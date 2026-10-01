from env_client import make_env
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import itertools

def trial(pair):
 q2,q4=pair;e=make_env();s,i=e.reset(seed=1);ro=s.get_object_from_name('robot');cu=s.get_object_from_name('cube_0')
 fs=['pos_arm_joint'+str(j) for j in range(1,8)]
 goal=np.array([0,q2,np.pi,q4,0,-.873,np.pi/2]);orig=np.array([s.get(cu,f) for f in ['x','y','z']]);maxerr=0
 for k in range(115):
  q=np.array([s.get(ro,f) for f in fs]);a=np.zeros(11,dtype=np.float32);a[3:10]=np.clip((goal-q)*2,-.1,.1);a[10]=1
  if k>85:
   a[0]=np.clip((orig[0]-s.get(ro,'pos_base_x'))*1.1,-.1,.1)
  s,r,t,tr,i=e.step(a)
  if t:break
 reached=[round(float(s.get(ro,f)),3) for f in fs]
 for k in range(50):
  q=np.array([s.get(ro,f) for f in fs]);a=np.zeros(11,dtype=np.float32);a[3:10]=np.clip((goal-q)*2,-.1,.1);a[10]=1;a[1]=-.045;a[0]=np.clip((orig[0]-s.get(ro,'pos_base_x'))*1.1,-.1,.1)
  s,r,t,tr,i=e.step(a)
  cur=np.array([s.get(cu,f) for f in ['x','y','z']]);maxerr=max(maxerr,float(np.linalg.norm(cur-orig)))
  if t:break
 print('RESULT',pair,'REACHED',reached,'CUBE',np.round(cur,3).tolist(),'MOVE',round(maxerr,3),'REWARD',r,'TERM',t,flush=True)
 e.close()
with ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(trial,itertools.product([-1.8,-1.2,-.6,0,.6,1.2],[-2.55,-1.5,-.5])))

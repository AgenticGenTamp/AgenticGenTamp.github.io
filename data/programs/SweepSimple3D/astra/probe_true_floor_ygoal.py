from env_client import make_env
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from threading import Event
stop=Event()
def trial(pair):
 if stop.is_set():return
 q2,q4,q6=pair;e=make_env();s,i=e.reset(seed=1);ro=s.get_object_from_name('robot');cu=s.get_object_from_name('cube_0');wi=s.get_object_from_name('wiper_0');fs=['pos_arm_joint'+str(j) for j in range(1,8)]
 home=np.array([0,-.349,np.pi,-2.548,0,-.873,np.pi/2]);goal=home.copy();orig=np.array([s.get(cu,f) for f in ['x','y','z']]);wo=np.array([s.get(wi,f) for f in ['x','y']]);maxdirect=0;old=orig.copy()
 for k in range(420):
  if k==60:goal=np.array([0,q2,np.pi,q4,0,q6,np.pi/2])
  q=np.array([s.get(ro,f) for f in fs]);a=np.zeros(11,dtype=np.float32);a[3:10]=np.clip((goal-q)*2,-.1,.1);a[10]=1;a[2]=np.clip((np.pi/2-s.get(ro,'pos_base_rot'))*1.1,-.1,.1);a[0]=np.clip((s.get(cu,'x')-s.get(ro,'pos_base_x'))*1.1,-.1,.1)
  if k<60:a[1]=np.clip((-.5-s.get(ro,'pos_base_y'))*1.1,-.1,.1)
  if k>=215:a[1]=.025
  s,r,t,tr,i=e.step(a);cur=np.array([s.get(cu,f) for f in ['x','y','z']]);w=np.array([s.get(wi,f) for f in ['x','y']])
  if np.linalg.norm(w-wo)<.02:maxdirect=max(maxdirect,float(np.linalg.norm(cur-orig)))
  if np.linalg.norm(cur-old)>.025 or k in [214,294,419]:
   print(pair,k,'BASE',np.round([s.get(ro,'pos_base_x'),s.get(ro,'pos_base_y')],3).tolist(),'Q',np.round(q[[1,3,5]],3).tolist(),'CUBE',np.round(cur,3).tolist(),'WIPER',np.round(w,3).tolist(),'R',r,'TERM',t,flush=True);old=cur.copy()
  if t:break
 print('FINAL',pair,'DIRECT',maxdirect,'MOVE',float(np.linalg.norm(cur-orig)),flush=True)
 if float(np.linalg.norm(cur-orig))>.18:stop.set()
 e.close()
pairs=[(1.65049,-1.23189,-.25921)]
with ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(trial,pairs))

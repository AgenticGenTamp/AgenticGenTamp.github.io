from env_client import make_env
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def run(goalname):
 e=make_env();s,i=e.reset(seed=1,options={'object_count':1})
 def v(n,fs):
  o=s.get_object_from_name(n);return np.array([float(s.get(o,f)) for f in fs])
 c0=v('cube_0',['x','y','z']);base=np.array([c0[0],c0[1]+.85,-np.pi/2]);q=np.array([0,2.4,np.pi,.3,0,-.873,np.pi/2]);goal=np.array([.5,0.]) if goalname=='island' else np.array([2.62,.625]);phase=0;release=None
 for k in range(650):
  actual=v('robot',['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint'+str(j) for j in range(1,8)]);c=v('cube_0',['x','y','z'])
  if k>=160:q[1]=1.2
  if k==210:base=np.array([1.55,.1,-np.pi/2]) if goalname=='island' else np.array([1.55,.625,-np.pi/2]);phase=1
  if phase==1 and np.linalg.norm(actual[:2]-base[:2])<.015:base[2]=-np.pi if goalname=='island' else 0;phase=2
  yawerr=(base[2]-actual[2]+np.pi)%(2*np.pi)-np.pi
  if phase==2 and abs(yawerr)<.01:phase=3
  if phase==3:
   base[:2]=actual[:2]+goal-c[:2]
   if np.linalg.norm(goal-c[:2])<.004:release=k;phase=4
  a=np.zeros(11,dtype=np.float32);err=base-actual[:3];err[2]=yawerr
  a[:3]=np.clip(err,-.025,.025);a[3:10]=np.clip(q-actual[3:],-.1,.1);a[10]=int(k>=150 and release is None)
  s,r,t,tr,i=e.step(a)
  if k%20==0 or r!=-1 or t or phase==4:print('GOAL',goalname,'STEP',k,'PHASE',phase,'BASE',np.round(actual[:3],3),'CUBE',np.round(v('cube_0',['x','y','z']),4),'R',r,'T',t,flush=True)
  if t or tr or release is not None and k>release+30:break
 e.close()
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,['island','cooking']))

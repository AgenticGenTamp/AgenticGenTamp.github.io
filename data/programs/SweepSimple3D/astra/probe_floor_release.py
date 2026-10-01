from env_client import make_env
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def run(goal):
 e=make_env();s,i=e.reset(seed=1,options={'object_count':1})
 def v(n,fs):
  o=s.get_object_from_name(n);return np.array([float(s.get(o,f)) for f in fs])
 c0=v('cube_0',['x','y','z']);up=goal[1]>1;theta=np.pi/2 if up else -np.pi/2;base=np.array([c0[0],c0[1]-.85 if up else c0[1]+.85,theta]);q=np.array([0,2.4,np.pi,.3,0,-.873,np.pi/2]);release=None
 for k in range(650):
  actual=v('robot',['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint'+str(j) for j in range(1,8)]);c=v('cube_0',['x','y','z'])
  if k>=170 and release is None:
   base[:2]=actual[:2]+np.clip(np.array(goal)-c[:2],-.015,.015)
   if np.linalg.norm(np.array(goal)-c[:2])<.008:release=k
  a=np.zeros(11,dtype=np.float32);err=base-actual[:3];err[2]=(err[2]+np.pi)%(2*np.pi)-np.pi
  a[:3]=np.clip(err,-.04 if k<150 else -.015,.04 if k<150 else .015);a[3:10]=np.clip(q-actual[3:],-.1,.1);a[10]=int(k>=150 and release is None)
  s,r,t,tr,i=e.step(a)
  if k%40==0 or r!=-1 or t or release is not None:print('GOAL',goal,'STEP',k,'BASE',np.round(actual[:3],3),'CUBE',np.round(v('cube_0',['x','y','z']),4),'R',r,'T',t,flush=True)
  if t or tr or release is not None and k>release+20:break
 e.close()
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,[(2.1,.625),(1.1,0),(1.185,2.1),(1.5,1.5)]))

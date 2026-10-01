from env_client import make_env
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def run(seed):
 e=make_env();s,i=e.reset(seed=seed,options={'object_count':1})
 def v(n,fs):
  o=s.get_object_from_name(n);return np.array([float(s.get(o,f)) for f in fs])
 c0=v('cube_0',['x','y','z']);base=[c0[0],c0[1]+.85,-np.pi/2];q=np.array([0,2.4,np.pi,.3,0,-.873,np.pi/2])
 for k in range(200):
  actual=v('robot',['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint'+str(j) for j in range(1,8)])
  a=np.zeros(11,dtype=np.float32);a[:3]=np.clip(np.array(base)-actual[:3],-.07,.07)
  if k>=160:q[1]=1.2
  a[3:10]=np.clip(q-actual[3:],-.1,.1);a[10]=int(k>=150)
  s,r,t,tr,i=e.step(a)
  if k in [149,159,199]:print('SEED',seed,'STEP',k,'CUBE',np.round(v('cube_0',['x','y','z']),4),'Q',np.round(v('robot',['pos_arm_joint'+str(j) for j in range(1,8)]),4),flush=True)
 e.close()
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,[0,1,2,3,4,5]))

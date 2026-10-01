from env_client import make_env
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def run(goal):
 e=make_env();s,i=e.reset(seed=1,options={'object_count':1})
 def v(n,fs):
  o=s.get_object_from_name(n);return np.array([float(s.get(o,f)) for f in fs])
 c0=v('cube_0',['x','y','z']);base=np.array([c0[0],c0[1]+.85,-np.pi/2]);q=np.array([0,2.4,np.pi,.3,0,-.873,np.pi/2])
 for k in range(650):
  actual=v('robot',['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint'+str(j) for j in range(1,8)])
  if k>=160:q[1]=1.2
  if k==210:base=np.array([1.5,.6,0]) if goal=='cooking' else np.array([1.2,1.5,np.pi/2])
  a=np.zeros(11,dtype=np.float32);err=base-actual[:3];err[2]=(err[2]+np.pi)%(2*np.pi)-np.pi
  a[:3]=np.clip(err,-.025,.025);a[3:10]=np.clip(q-actual[3:],-.1,.1);a[10]=int(150<=k<500)
  s,r,t,tr,i=e.step(a)
  if k%25==0 or r!=-1 or t:print('GOAL',goal,'STEP',k,'BASE',np.round(actual[:3],3),'CUBE',np.round(v('cube_0',['x','y','z']),3),'R',r,'T',t,flush=True)
  if t or tr:break
 e.close()
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,['cooking','leftside']))

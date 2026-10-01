from env_client import make_env
import numpy as np
E=make_env();s,i=E.reset(seed=1,options={'object_count':1})
def v(n,fs):
 o=s.get_object_from_name(n);return np.array([float(s.get(o,f)) for f in fs])
c0=v('cube_0',['x','y','z']);base=np.array([c0[0],c0[1]+.85,-np.pi/2]);q=np.array([0,2.4,np.pi,.3,0,-.873,np.pi/2]);release=None;phase=0;goal=np.array([2.1,.625])
for k in range(650):
 actual=v('robot',['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint'+str(j) for j in range(1,8)]);c=v('cube_0',['x','y','z'])
 if k==170:base[:2]=[1.25,1.4];phase=1
 if phase==1 and np.linalg.norm(actual[:2]-base[:2])<.015:base[2]=0;phase=2
 yawerr=(base[2]-actual[2]+np.pi)%(2*np.pi)-np.pi
 if phase==2 and abs(yawerr)<.01:phase=3
 if phase==3 and release is None:
  base[:2]=actual[:2]+np.clip(goal-c[:2],-.015,.015)
  if np.linalg.norm(goal-c[:2])<.008:release=k
 a=np.zeros(11,dtype=np.float32);err=base-actual[:3];err[2]=yawerr
 a[:3]=np.clip(err,-.025 if k<150 else -.015,.025 if k<150 else .015);a[3:10]=np.clip(q-actual[3:],-.1,.1);a[10]=int(k>=150 and release is None)
 s,r,t,tr,i=E.step(a)
 if k%20==0 or r!=-1 or t or release is not None:print(k,phase,'BASE',np.round(actual[:3],3),'CUBE',np.round(v('cube_0',['x','y','z']),4),'R',r,'T',t,flush=True)
 if t or tr or release is not None and k>release+20:break
E.close()

from env_client import make_env
import numpy as np
E=make_env();s,i=E.reset(seed=1);r=s.get_object_from_name('robot');c=s.get_object_from_name('cube_0');xy=np.array([s.get(c,f) for f in ('x','y')]);q=np.array([0,1.65049089,np.pi,-1.23189293,0,-.25920883,np.pi/2]);base=np.array([xy[0],xy[1]+.8,-np.pi/2])
for k in range(500):
 b=np.array([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']]);j=np.array([s.get(r,'pos_arm_joint%d'%j) for j in range(1,8)]);a=np.zeros(11);a[3:10]=np.clip(q-j,-.1,.1);a[10]=1
 if k<120:a[:3]=np.clip(base-b,-.06,.06)
 else:
  a[0]=np.clip(s.get(c,'x')-b[0],-.035,.035);a[1]=-.025;a[2]=np.clip(-np.pi/2-b[2],-.05,.05)
 s,re,t,tr,i=E.step(a)
 if k%20==0 or t:print(k,'BASE',np.round(b,3),'CUBE',[round(s.get(c,f),3) for f in ['x','y','z']],'R',re,'DONE',t,flush=True)
 if t:break
E.close()

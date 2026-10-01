from env_client import make_env
import numpy as np

e=make_env();s,i=e.reset(seed=1);ro=s.get_object_from_name('robot');cu=s.get_object_from_name('cube_0');wi=s.get_object_from_name('wiper_0');fs=['pos_arm_joint'+str(j) for j in range(1,8)]
goal=np.array([0,2.4,np.pi,-.5,0,-.873,np.pi/2]);orig=np.array([s.get(cu,f) for f in ['x','y','z']]);old=orig.copy()
for k in range(295):
 q=np.array([s.get(ro,f) for f in fs]);a=np.zeros(11,dtype=np.float32);a[3:10]=np.clip((goal-q)*2,-.1,.1);a[10]=1
 if k>85:a[0]=np.clip((orig[0]-s.get(ro,'pos_base_x'))*1.1,-.1,.1)
 if 115<=k<165:a[1]=-.045
 if 165<=k<210:a[1]=.045
 if k>=210:a[1]=-.045
 s,r,t,tr,i=e.step(a);cur=np.array([s.get(cu,f) for f in ['x','y','z']])
 if np.linalg.norm(cur-old)>.003 or k%10==0:
  print(k,'BASE',np.round([s.get(ro,'pos_base_x'),s.get(ro,'pos_base_y')],3).tolist(),'Q',np.round(q,3).tolist(),'CUBE',np.round(cur,4).tolist(),'WIPER',np.round([s.get(wi,f) for f in ['x','y','z','qw','qx','qy','qz']],3).tolist(),'R',r,'TERM',t,flush=True);old=cur.copy()
 if t:break
e.close()

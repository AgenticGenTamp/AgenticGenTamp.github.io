from env_client import make_env
from approach import GeneratedApproach
import numpy as np
from scipy.spatial.transform import Rotation
e=make_env()
for j in range(7):
 ds=[];rs=[]
 for sg in [-1,1]:
  s,i=e.reset(seed=0);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
  while True:
   s,*_=e.step(p.get_action(s))
   if p.stage==3 and p.at(s,p.target(p.blocker+.36*p.out)):break
  b=np.array([p.g(s,'blocker','pose_'+x) for x in 'xyz'])
  a=np.zeros(11,np.float32);a[3+j]=sg*.03;s,*_=e.step(a)
  d=np.array([p.g(s,'blocker','pose_'+x) for x in 'xyz'])-b;ds.append(d)
  rs.append(Rotation.from_quat([p.g(s,'blocker','pose_q'+x) for x in 'xyzw']))
 rv=(rs[1]*rs[0].inv()).as_rotvec()/.06
 print(j+1,np.round((ds[1]-ds[0])/.06,4),np.round(rv,4))
e.close()

from probe_tilt import pose
from kinematics import HOME
from env_client import make_env
import numpy as np
import time

e=make_env();s,inf=e.reset(seed=42);r=s.get_object_from_name('robot');o=s.get_object_from_name('cube1');b0=np.array([s.get(o,'x')-.625,s.get(o,'y')-.001,0]);steps=0

def move(q,b,g,n):
 global s,steps
 for i in range(n):
  cur=np.array([s.get(r,'pos_arm_joint'+str(j)) for j in range(1,8)]);base=np.array([s.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
  a=np.r_[np.clip(b-base,-.1,.1),np.clip(2*(q-cur),-.1,.1),g].astype(np.float32)
  s,rew,t,tr,inf=e.step(a);steps+=1
  if t or rew!=-1:
   print('FOUND',steps,'reward',rew,'term',t,'cube',[s.get(o,f) for f in ['x','y','z']],flush=True)
   if t: raise SystemExit
 return np.round([s.get(o,f) for f in ['x','y','z']],4)

move(HOME,b0,0,35);down=pose(.5,-.04);move(down,b0,0,95);move(down,b0,1,15)
q=pose(.6,.245,-np.pi/2);print('lift',move(q,b0,1,100),flush=True)
b=np.array([.775,-.001,0.]);print('insert',move(q,b,1,80),flush=True)
for zi,z in enumerate([.22,.26,.3,.18,.14]):
 q=pose(.6,z-.095,-np.pi/2)
 print('z',z,'at',move(q,b,1,30),flush=True)
 for yi,y in enumerate(np.arange(-.2,.201,.04)):
  xs=np.arange(1.4,1.651,.04)
  if yi%2:xs=xs[::-1]
  for x in xs:
   b=np.array([x-.765,y-.001,0.]);p=move(q,b,1,5)
  print('row',z,y,'at',p,flush=True)
# Release on side after scan
b=np.array([1.5-.765,-.15-.001,0.]);q=pose(.6,.12,-np.pi/2)
print('release-left',move(q,b,1,30),move(q,b,0,30),flush=True)
e.close()

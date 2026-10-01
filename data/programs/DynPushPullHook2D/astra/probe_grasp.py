import numpy as np,math
from env_client import make_env

def run(seed, offset):
 e=make_env();s,i=e.reset(seed=seed)
 rb=s.get_objects(e.observation_space.get_type('kin_robot'))[0];h=s.get_objects(e.observation_space.get_type('hook'))[0]
 hx,hy=s.get(h,'x'),s.get(h,'y')
 for k in range(100):
  dx=hx-offset-s.get(rb,'x');dy=hy-s.get(rb,'y');dt=-s.get(rb,'theta')
  a=np.array([np.clip(dx,-.049,.049),np.clip(dy,-.049,.049),np.clip(dt,-.064,.064),0,.019],dtype=np.float32)
  s,r,d,tr,i=e.step(a)
  if max(abs(dx),abs(dy),abs(dt))<.001:break
 for k in range(14):
  s,r,d,tr,i=e.step(np.array([0,0,0,0,-.019],dtype=np.float32))
 print(seed,offset,'base',s.get(rb,'x'),s.get(rb,'y'),'hook',s.get(h,'x'),s.get(h,'y'),'held',s.get(h,'held'),flush=True)
 if s.get(h,'held'):
  for k in range(10):s,r,d,tr,i=e.step(np.array([-.049,0,0,0,0],dtype=np.float32))
  print('MOVED',s.get(h,'x'),s.get(h,'y'),s.get(h,'held'),flush=True)
 e.close()
for offset in [.24,.34,.44,.54,.64,.74,.84,.94,1.04,1.14]:run(0,offset)

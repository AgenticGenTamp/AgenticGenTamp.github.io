import math
import numpy as np
from env_client import make_env

for seed in (0,1,2,3,4):
 e=make_env(); s,_=e.reset(seed=seed); typ=e.observation_space.get_type
 r=s.get_objects(typ('kin_robot'))[0]; b=s.get_objects(typ('target_block'))[0]
 # Rotate horizontal and place the jaw axis at the block center without advancing.
 for i in range(20):
  th=float(s.get(r,'theta')); by=float(s.get(b,'y'))
  a=np.array([0,np.clip(by-float(s.get(r,'y')),-.03,.03),np.clip(-th,-.19,.19),0,0],float)
  s,*_=e.step(a)
 bx0=float(s.get(b,'x'))
 contact=None
 for i in range(80):
  s,*_=e.step(np.array([.01,0,0,0,0],float))
  if abs(float(s.get(b,'x'))-bx0)>.001:
   contact=i; break
 for j in range(15):
  s,*_=e.step(np.array([.008,0,0,0,-.019],float))
 for j in range(12): s,*_=e.step(np.array([0,.02,0,0,0],float))
 print(seed,'contact',contact,'sep',round(float(s.get(b,'x'))-float(s.get(r,'x')),3),
       'gap',round(float(s.get(r,'finger_gap')),3),'held',float(s.get(b,'held')),flush=True)
 e.close()

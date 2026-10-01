from env_client import make_env
import numpy as np
import math
offset=.50
for seed in range(12):
 e=make_env();s,_=e.reset(seed=seed);ro=s.get_object_from_name('robot');bl=s.get_object_from_name('target_block')
 def g(o,f): return s.get(o,f)
 def go(x,y,th):
  global s
  for _ in range(80):
   d=np.array([x-g(ro,'x'),y-g(ro,'y'),th-g(ro,'theta'),.24-g(ro,'arm_joint'),.32-g(ro,'finger_gap')]);a=np.clip(d,[-.049,-.049,-.19,-.099,-.019],[.049,.049,.19,.099,.019]);s,*_=e.step(a)
   if max(abs(d[:3]))<.0002:break
 go(g(ro,'x'),1.72,g(ro,'theta'));go(g(bl,'x'),1.72,-math.pi/2);go(g(bl,'x'),g(bl,'y')+g(bl,'height')/2+.50,-math.pi/2)
 for k in range(50):
  s,*_=e.step(np.array([np.clip(g(bl,'x')-g(ro,'x'),-.02,.02),-.008,0,0,-.019]))
  if g(bl,'held') or g(ro,'y')<.5:break
 print('seed',seed,'width',g(bl,'width'),'height',g(bl,'height'),'offset',offset,'held',g(bl,'held'),'block',g(bl,'x'),g(bl,'y'),g(bl,'theta'),'base',g(ro,'x'),g(ro,'y'),'gap',g(ro,'finger_gap'))
 for k in range(6):s,*_=e.step(np.array([0.,.049,0,0,0]))
 print('lifted',g(bl,'held'),g(bl,'y'))
 for k in range(3):
  s,*_=e.step(np.array([0.,0,0,0,.001]))
  print('open',k,g(bl,'held'),g(ro,'finger_gap'))
 e.close()

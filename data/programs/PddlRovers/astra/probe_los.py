from env_client import make_env
import numpy as np

for y in [-1.9,-1.95,-2.,-2.01,-2.05,-2.1]:
 e=make_env();s,_=e.reset(seed=0);r=s.get_object_from_name('rover0')
 def act(dx=0,dy=0,op=0):
  global s
  a=np.zeros(8);a[:2]=dx,dy;a[3]=op;s,*_=e.step(a)
 for _ in range(11):act(dy=.2)
 act(op=-.5);act(op=-1/6)
 while s.get(r,'y')>y+.0001:act(dy=max(-.2,y-s.get(r,'y')))
 act(op=.5)
 print(y,s.get(r,'x'),s.get(r,'y'),s.get(s.get_object_from_name('objective0'),'received_image'),flush=True)
 e.close()

from env_client import make_env
import numpy as np

e=make_env();s,_=e.reset(seed=0);r=s.get_object_from_name('rover0')
def step(dx=0,dy=0,op=0):
 global s
 a=np.zeros(8);a[:2]=(dx,dy);a[3]=op;s,*_=e.step(a)
def report(label):
 print(label,[s.get(r,x) for x in ['x','y','calibrated']],[(o.name,s.get(o,'have_image_rover0'),s.get(o,'received_image')) for o in s.get_objects(e.observation_space.get_type('objective'))],flush=True)
for _ in range(11):step(dy=.2)
step(op=-.5);step(op=-1/6);report('image at .45')
for _ in range(11):step(dy=-.2)
step(op=.5);report('send home')
for _ in range(2):step(dy=-.2)
step(op=.5);report('send bottom')
for _ in range(3):step(dx=-.2)
for yy in [-2.25,-2.2,-2.,-1.5,0.,1.,1.5]:
 for _ in range(30):
  dy=np.clip(yy-s.get(r,'y'),-.2,.2)
  if abs(dy)<1e-6:break
  old=s.get(r,'y');step(dy=dy)
  if abs(s.get(r,'y')-old)<1e-6:break
 step(dx=.26-s.get(r,'x'));report('wall .26 '+str(yy))
 step(dx=.25-s.get(r,'x'));report('wall .25 '+str(yy))
 step(dx=.4-s.get(r,'x'))
e.close()

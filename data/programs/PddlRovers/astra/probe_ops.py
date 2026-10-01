from env_client import make_env
import numpy as np
E=make_env();s,_=E.reset(seed=0)
r=s.get_object_from_name('rover0')
def step(dx,dy,op=0):
 global s
 a=np.zeros(8);a[:2]=dx,dy;a[3]=op;s,_,t,tr,i=E.step(a)
 print('pos',round(s.get(r,'x'),3),round(s.get(r,'y'),3),'cal',s.get(r,'calibrated'),'imgs',[(o.name,s.get(o,'have_image_rover0')) for o in s.get_objects(E.observation_space.get_type('objective'))])
for _ in range(17):step(0,.2,-.5)
for _ in range(4):step(0,0,-1/6);step(0,0,-.5)
E.close()

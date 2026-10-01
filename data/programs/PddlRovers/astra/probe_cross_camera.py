from env_client import make_env
import numpy as np
for seed in [0,1,3,4]:
 e=make_env();s,_=e.reset(seed=seed,options={'object_count':4});objs=s.get_objects(e.observation_space.get_type('objective'));r=s.get_object_from_name('rover0')
 print('seed',seed,[(o.name,round(s.get(o,'x'),2),round(s.get(o,'y'),2)) for o in objs])
 for t in range(15):
  a=np.zeros(8);a[1]=.2;s,*_=e.step(a)
 for t in range(12):
  a=np.zeros(8);a[3]=-.5 if t%2==0 else -1/6;s,*_=e.step(a)
 print('pose',s.get(r,'x'),s.get(r,'y'),'images',[(o.name,s.get(o,'have_image_rover0')) for o in objs]);e.close()

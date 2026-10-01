from env_client import make_env
import numpy as np
for target_y in [.25,.45,.65,.85]:
 e=make_env();s,_=e.reset(seed=0,options={'object_count':4});r=s.get_object_from_name('rover0')
 for _ in range(4):
  a=np.zeros(8);a[0]=max(-.2,.32-s.get(r,'x'));s,*_=e.step(a)
 for _ in range(15):
  a=np.zeros(8);a[1]=min(.2,target_y-s.get(r,'y'));s,*_=e.step(a)
 for t in range(10):
  a=np.zeros(8);a[3]=-.5 if t%2==0 else -1/6;s,*_=e.step(a)
 print('pose',s.get(r,'x'),s.get(r,'y'),'images',[(o.name,s.get(o,'have_image_rover0')) for o in s.get_objects(e.observation_space.get_type('objective'))]);e.close()

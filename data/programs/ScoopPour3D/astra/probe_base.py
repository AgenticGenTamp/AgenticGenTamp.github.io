from env_client import make_env
import numpy as np
E=make_env();s,_=E.reset(seed=0)
for k in range(35):
 a=np.zeros(11); a[0]=.05 if k<12 else 0; a[1]=-.05 if k<6 else (.05 if k<28 else 0)
 s,r,t,tr,i=E.step(a)
 if k%3==0:
  print(k,r,[(n,[round(s.get(s.get_object_from_name(n),f),3) for f in fs]) for n,fs in [('robot',['pos_base_x','pos_base_y','pos_base_rot']),('bin_yellow_0',['x','y','z']),('bin_green_0',['x','y','z']),('scoop_0',['x','y','z']),('cube_0',['x','y','z'])]])
E.close()

from env_client import make_env
import numpy as np

e=make_env(); s,i=e.reset(seed=0)
print('INFO',i,'MAX',e.max_steps)
for n in sorted(s.get_object_names()):
 o=s.get_object_from_name(n); print(n,{f:round(s.get(o,f),5) for f in e.observation_space.type_features[o.type]})
for k in range(6):
 a=np.zeros(18); a[0]=.1; a[3]=.1
 s,r,t,tr,i=e.step(a)
 o=s.get_object_from_name('robot');print(k,r,t,tr,i,{f:round(s.get(o,f),4) for f in e.observation_space.type_features[o.type]})
e.close()

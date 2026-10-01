from env_client import make_env
import numpy as np
E=make_env(); s,i=E.reset(seed=0)
print('INFO',i,'MAX',E.max_steps)
for n in sorted(s.get_object_names()):
 o=s.get_object_from_name(n)
 print(n, {f:round(s.get(o,f),4) for f in E.observation_space.type_features[o.type]})
for k in range(3):
 s,r,t,tr,i=E.step(np.zeros(11)); print('STEP',k,r,t,tr,i)
E.close()

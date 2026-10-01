from env_client import make_env
import numpy as np
E=make_env(); s,info=E.reset(seed=0)
print('INFO',info,flush=True)
for t in E.observation_space.types:
 for o in s.get_objects(t):
  print(o, {f:round(s.get(o,f),4) for f in E.observation_space.type_features[o.type]},flush=True)
for i in range(3):
 s,r,d,tr,inf=E.step(np.zeros(11,dtype=np.float32));print('STEP',i,r,d,tr,inf,flush=True)
E.close()

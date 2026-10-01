from env_client import make_env
import numpy as np
with make_env() as e:
 for seed in [0,1,2]:
  s,info=e.reset(seed=seed)
  print('SEED',seed,'INFO',info,flush=True)
  for name in sorted(s.get_object_names()):
   obj=s.get_object_from_name(name)
   fs=e.observation_space.type_features[obj.type]
   print(name, obj.type, {f:round(s.get(obj,f),3) for f in fs},flush=True)
  a=np.zeros(11)
  s,r,t,tr,i=e.step(a)
  print('STEP',r,t,tr,i,flush=True)

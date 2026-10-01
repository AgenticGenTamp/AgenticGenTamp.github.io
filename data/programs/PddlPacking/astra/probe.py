from env_client import make_env
import numpy as np
e=make_env(); s,info=e.reset(seed=0)
print('info',info)
for t in e.observation_space.types:
 for o in s.get_objects(t):
  print(o.name,{f:round(s.get(o,f),5) for f in e.observation_space.type_features[t]})
e.close()

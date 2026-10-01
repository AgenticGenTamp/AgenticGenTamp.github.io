from env_client import make_env
import numpy as np
E=make_env()
s,i=E.reset(seed=0)
print('INFO',i,'MAX',E.max_steps)
for typ in E.observation_space.types:
 for o in s.get_objects(typ):
  print(o.name, {f:round(float(s.get(o,f)),5) for f in E.observation_space.type_features[typ]})
E.close()

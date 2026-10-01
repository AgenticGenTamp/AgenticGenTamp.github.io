from env_client import make_env
import numpy as np
e=make_env();s,i=e.reset(seed=0)
print('INFO',i)
for t in e.observation_space.types:
 for o in s.get_objects(t):
  print(o.name,t.name,dict(zip(e.observation_space.type_features[t],[s.get(o,f) for f in e.observation_space.type_features[t]])))
e.close()

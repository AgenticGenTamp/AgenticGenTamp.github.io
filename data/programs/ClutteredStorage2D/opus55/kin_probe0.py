import numpy as np
from env_client import make_env
env=make_env()
obs,info=env.reset(seed=0)
print(env.observation_space.type_features)
for n in obs.get_object_names():
    o=obs.get_object_from_name(n)
    print(n,o.type,{f:round(float(obs.get(o,f)),4) for f in env.observation_space.type_features[o.type]})
print(env.action_space)

from env_client import make_env
import numpy as np, math
env=make_env(); T=env.observation_space.get_type
f=env.observation_space.type_features[T("robot")]
o,info=env.reset(seed=27)
r=o.get_objects(T("robot"))[0]; print(dict(zip(f,np.round([o.get(r,x) for x in f],4))))
for p in o.get_objects(T("surface")): print(p.name,[round(float(o.get(p,k)),3) for k in env.observation_space.type_features[T("surface")]])

from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
print(type(obs), info)
for t in obs.get_objects.__self__.__class__.__mro__: pass
for name in obs.get_object_names():
    o = obs.get_object_from_name(name)
    print(name, o.type if hasattr(o,'type') else '', end=': ')
    try:
        feats = env.observation_space.get_type(o.type.name if hasattr(o.type,'name') else o.type)
    except Exception as e: feats=None
    print(o)
print(dir(obs))

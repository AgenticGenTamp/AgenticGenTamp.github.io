from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
print("info", info)
tf = env.observation_space.type_features
print({k if isinstance(k,str) else k.name: v for k,v in tf.items()})
for name in sorted(obs.get_object_names()):
    o = obs.get_object_from_name(name)
    feats = tf[o.type]
    print(name, o.type.name, {f: round(float(obs.get(o,f)),4) for f in feats})
print("max_steps", env.max_steps)

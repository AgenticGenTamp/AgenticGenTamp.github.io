import numpy as np
from env_client import make_env
env = make_env()
obs, info = env.reset(seed=0)
print("info:", info)
print("names:", sorted(obs.get_object_names()))
for o in obs:
    print(o.name, o.type.name, dict(zip(obs.type_features[o.type], np.round(obs.data[o],4))))

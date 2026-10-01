from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
print("info", info)
names = sorted(obs.get_object_names())
print(len(names), names)
for o in sorted(obs.data, key=lambda o:o.name):
    print(o.name, o.type.name, np.round(obs.data[o],3).tolist())
env.close()

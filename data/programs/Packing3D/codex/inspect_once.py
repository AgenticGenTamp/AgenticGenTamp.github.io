from env_client import make_env
import numpy as np

env = make_env()
obs, info = env.reset(seed=0, options={"object_count": 3})
print("info", info, "max", env.max_steps)
print("names", sorted(obs.get_object_names()))
for name in sorted(obs.get_object_names()):
    obj = obs.get_object_from_name(name)
    fs = obs.type_features[obj.type]
    print(name, obj.type.name, dict(zip(fs, obs.data[obj].tolist())))
print("low", env.action_space.low, "high", env.action_space.high)
env.close()

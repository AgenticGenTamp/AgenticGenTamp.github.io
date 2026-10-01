from env_client import make_env
import numpy as np
env = make_env()
tf = {k.name:v for k,v in env.observation_space.type_features.items()}
for seed in [1,2,3,7,11]:
    obs, info = env.reset(seed=seed)
    names=[n for n in obs.get_object_names() if n.startswith('cube')]
    print(seed, info, sorted(names))
env.close()

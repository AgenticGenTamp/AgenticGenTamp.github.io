from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
for n in sorted(obs.get_object_names()):
    o = obs.get_object_from_name(n)
    if o.type.name in ('mujoco_fixture','mujoco_drawer','mujoco_tidybot_robot'):
        print(n, {f: round(float(obs.get(o,f)),3) for f in obs.type_features[o.type] if not f.startswith('vel')})

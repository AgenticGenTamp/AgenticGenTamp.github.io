from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
for t, objs in [(t, obs.get_objects(t)) for t in set(o.type for o in obs.data)]:
    print("TYPE", t, [o.name for o in objs][:6])
    print("  feats", obs.type_features[t])
r = obs.get_object_from_name("robot")
print("robot feats", list(zip(obs.type_features[r.type], obs.data[r])))
print("action space", env.action_space if hasattr(env,'action_space') else None)
env.close()

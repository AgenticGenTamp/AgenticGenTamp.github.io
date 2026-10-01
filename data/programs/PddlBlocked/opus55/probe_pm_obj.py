from env_client import make_env
import numpy as np
env=make_env(); obs,_=env.reset(seed=0)
for o in obs.get_objects(None) if False else []: pass
T={t.name:t for t in env.observation_space.type_features}
for tn,t in T.items():
    for o in obs.get_objects(t):
        print(tn,o.name,[round(float(obs.get(o,f)),3) for f in ["pose_x","pose_y","pose_z"]] if tn!="robot" else "")

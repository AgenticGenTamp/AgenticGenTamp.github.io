from env_client import make_env
import numpy as np
env = make_env()
tf = {t.name:(t,f) for t,f in env.observation_space.type_features.items()}
def dump(obs):
    for name in ['robot','surface','block']:
        t,feats=tf[name]
        for o in obs.get_objects(t):
            print(name, o.name, [round(float(obs.get(o,f)),3) for f in feats])
for seed in [0,1,2,3]:
    obs, info = env.reset(seed=seed)
    print("seed", seed, "info", info)
    dump(obs)
env.close()

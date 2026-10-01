from env_client import make_env
import numpy as np
env = make_env()
for seed in range(4):
    obs, info = env.reset(seed=seed)
    names = sorted(obs.get_object_names())
    print("seed", seed, "objs", names)
    for n in names:
        o = obs.get_object_from_name(n)
        feats = {f: round(obs.get(o, f),4) for f in obs.type_features[o.type]} if hasattr(obs,'type_features') else None
        print("  ", n, o.type.name)
    # print raw
    for n in names:
        o = obs.get_object_from_name(n)
        print("  ", n, np.round(obs.vec([o]),4))
    print(" info", info)
env.close()

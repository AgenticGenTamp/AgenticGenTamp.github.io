import numpy as np
from env_client import make_env

env = make_env()
for seed in range(4):
    obs, info = env.reset(seed=seed)
    r = obs.data[obs.get_object_from_name("robot")]
    print("=== seed", seed, "info", info)
    print("  base", np.round(r[:3],3), "q", np.round(r[3:10],3), "grip", r[10], "ga", r[11])
    for o in obs:
        d = obs.data[o]
        if o.type.name == "block":
            print("  blk %-8s pos %s he %s" % (o.name, np.round(d[:3],3), np.round(d[8:11],3)))
        elif o.type.name == "surface":
            print("  srf %-10s pos %s he %s" % (o.name, np.round(d[:3],3), np.round(d[7:10],3)))
        elif o.type.name != "robot":
            print("  oth %-10s %s %s" % (o.name, o.type.name, np.round(d,3)))
env.close()

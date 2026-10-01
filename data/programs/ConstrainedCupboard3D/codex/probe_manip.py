import numpy as np
from env_client import make_env


def dump(seed=0):
    env = make_env()
    s, info = env.reset(seed=seed)
    print("seed", seed, "info", info, "max", env.max_steps)
    print("names", s.get_object_names())
    for typ in env.observation_space.types:
        objs = s.get_objects(typ)
        if objs:
            print("TYPE", typ.name if hasattr(typ, "name") else typ)
        for o in objs:
            fs = env.observation_space.type_features[typ]
            print(o.name, {f: round(float(s.get(o, f)), 4) for f in fs})
    print("action", env.action_space.low, env.action_space.high)
    env.close()


if __name__ == "__main__":
    for seed in range(3):
        dump(seed)

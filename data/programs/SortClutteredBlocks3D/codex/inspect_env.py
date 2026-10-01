from env_client import make_env
import numpy as np

for seed in range(3):
    env = make_env()
    obs, info = env.reset(seed=seed)
    print("SEED", seed, "max", env.max_steps, "info", info)
    print("names", obs.get_object_names())
    for name in obs.get_object_names():
        obj = obs.get_object_from_name(name)
        typ = getattr(obj, "type", None)
        vals = {f: round(float(obs.get(obj, f)), 4) for f in obs.type_features[obj.type]}
        print(name, typ, vals)
    a = np.zeros(env.action_space.shape, dtype=np.float32); a[-1] = 1
    _, r, term, trunc, inf = env.step(a)
    print("zero", r, term, trunc, inf)
    env.close()

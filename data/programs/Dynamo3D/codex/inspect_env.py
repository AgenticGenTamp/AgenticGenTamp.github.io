from env_client import make_env
import numpy as np

for seed in range(3):
    env = make_env()
    obs, info = env.reset(seed=seed)
    print("SEED", seed, "max", env.max_steps, "info", info)
    print("names", obs.get_object_names())
    for name in obs.get_object_names():
        obj = obs.get_object_from_name(name)
        vals = {}
        features = env.observation_space.type_features[obj.type]
        for f in features:
            try:
                vals[f] = round(float(obs.get(obj, f)), 3)
            except Exception:
                pass
        print(name, obj.type.name, vals)
    for action in [np.zeros(11, dtype=np.float32), np.array([.1,0,0]+[0]*8,dtype=np.float32)]:
        nxt, r, term, trunc, inf = env.step(action)
        rob = nxt.get_object_from_name("robot")
        print("step", action[:3], r, term, trunc, inf,
              [round(float(nxt.get(rob,f)),3) for f in ("pos_base_x","pos_base_y","pos_base_rot")])
    env.close()

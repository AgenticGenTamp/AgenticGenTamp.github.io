from env_client import make_env
import numpy as np
env = make_env()
for seed in range(3):
    obs, info = env.reset(seed=seed)
    print("seed", seed, "info", info)
    for t in env.observation_space.types:
        for o in obs.get_objects(t):
            print(" ", t.name if hasattr(t,'name') else t, o.name, {f: round(float(obs.get(o,f)),3) for f in env.observation_space.type_features[t]})
print(env.max_steps)
a = np.zeros(18, dtype=np.float32)
obs, r, term, trunc, info = env.step(a)
print(r, term, trunc, info)
env.close()

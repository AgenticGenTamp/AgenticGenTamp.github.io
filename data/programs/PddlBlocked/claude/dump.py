from env_client import make_env
import numpy as np
env = make_env()
for seed in [0,1,2,3]:
    obs, info = env.reset(seed=seed)
    print("=== seed", seed, "info", info)
    for o in sorted(obs.data, key=lambda x: x.name):
        print(f"{o.name:12s} {o.type.name:8s}", np.round(obs.data[o],4).tolist())
print("max_steps", env.max_steps)
print("types", [t.name for t in env.observation_space.types])
print(env.observation_space.type_features)
env.close()

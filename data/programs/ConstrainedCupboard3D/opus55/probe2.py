from env_client import make_env
import numpy as np, collections
env = make_env()
c = collections.Counter()
for seed in range(12, 60):
    obs, info = env.reset(seed=seed)
    fx = env.observation_space.get_type('mujoco_fixture')
    n = len(obs.get_objects(fx))
    c[(info['object_count'], n)] += 1
print(c)
for k in [1,4,5,6,8]:
    try:
        obs, info = env.reset(seed=0, options={'object_count':k})
        fx = env.observation_space.get_type('mujoco_fixture')
        print(k, info, len(obs.get_objects(fx)))
    except Exception as e: print(k, 'err', str(e)[:200])
env.close()

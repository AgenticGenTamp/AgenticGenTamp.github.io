from env_client import make_env
import numpy as np
env = make_env()
for seed in range(12):
    obs, info = env.reset(seed=seed)
    fx = env.observation_space.get_type('mujoco_fixture')
    mv = env.observation_space.get_type('mujoco_movable_object')
    cb = sorted([(o.name, round(obs.get(o,'x'),3), round(obs.get(o,'y'),3), round(obs.get(o,'z'),3)) for o in obs.get_objects(fx)])
    rods = sorted([(o.name, round(obs.get(o,'x'),2), round(obs.get(o,'y'),2)) for o in obs.get_objects(mv)])
    print(seed, info['object_count'], "ys", sorted(c[2] for c in cb), "xs", set(c[1] for c in cb), "zs", set(c[3] for c in cb))
    print("   rods", rods)
env.close()

from env_client import make_env
import numpy as np
env = make_env()
for seed in range(40):
    obs, info = env.reset(seed=seed)
    names = sorted(obs.get_object_names() if hasattr(obs,'get_object_names') else [])
    b = obs.get_object_from_name('bin_0'); 
    cs = [n for n in names if 'cube' in str(n)]
    s = f"seed {seed} n={info['object_count']} bin=({obs.get(b,'x'):.2f},{obs.get(b,'y'):.2f})"
    for c in obs.get_objects(env.observation_space.get_type('mujoco_movable_object')):
        if c.name.startswith('cube'):
            s += f" {c.name}=({obs.get(c,'x'):.2f},{obs.get(c,'y'):.2f})"
    print(s)
env.close()

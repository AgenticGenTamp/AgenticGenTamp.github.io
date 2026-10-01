from env_client import make_env
import numpy as np
env = make_env()
print(env.max_steps)
for seed in [0,1]:
    obs, info = env.reset(seed=seed)
    print("info", info)
    for t in obs.get_objects(env.observation_space.get_type('mujoco_movable_object')) if hasattr(env.observation_space,'get_type') else []:
        print(t.name, [round(float(obs.get(t,f)),3) for f in ['x','y','z','qw','qx','qy','qz','bb_x','bb_y','bb_z']])
    for n in obs.get_object_names():
        o = obs.get_object_from_name(n)
        if 'robot' in n:
            print(n, {f: round(float(obs.get(o,f)),3) for f in env.observation_space.type_features[o.type.name] if not f.startswith('vel')})
    if seed==0:
        for n in obs.get_object_names():
            o=obs.get_object_from_name(n)
            if o.type.name=='mujoco_fixture': print(n,[round(float(obs.get(o,f)),3) for f in ['x','y','z','qw','qx','qy','qz']])

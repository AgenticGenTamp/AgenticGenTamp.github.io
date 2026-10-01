from env_client import make_env
env=make_env()
for seed in [0,2,5,3]:
    obs,_=env.reset(seed=seed)
    for n in sorted(obs.get_object_names()):
        if n.startswith('cube'):
            o=obs.get_object_from_name(n); print(seed,n,[round(obs.get(o,f),3) for f in ['x','y','z','qw','qx','qy','qz','bb_x','bb_y','bb_z']])

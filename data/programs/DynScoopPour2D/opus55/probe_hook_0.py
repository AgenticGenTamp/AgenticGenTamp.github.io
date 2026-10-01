from env_client import make_env
env=make_env(); obs,info=env.reset(seed=0)
for n in obs.get_object_names():
    o=obs.get_object_from_name(n)
    fs=env.observation_space.type_features[o.type] if hasattr(env.observation_space,'type_features') else []
    print(n, o.type.name if hasattr(o.type,'name') else o.type, {f:round(float(obs.get(o,f)),3) for f in fs if f not in('vx','vy','omega','color_r','color_g','color_b') and not f.startswith('v') and not f.startswith('omega')})
env.close()

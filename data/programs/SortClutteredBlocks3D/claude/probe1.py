from env_client import make_env
env = make_env()
tf = {k.name:v for k,v in env.observation_space.type_features.items()}
obs, info = env.reset(seed=0)
print("info", info)
for name in sorted(obs.get_object_names()):
    o = obs.get_object_from_name(name)
    fs = tf[o.type.name]
    print(name, o.type.name, [f"{f}={float(obs.get(o,f)):.3f}" for f in fs])
print("max_steps", env.max_steps)
env.close()

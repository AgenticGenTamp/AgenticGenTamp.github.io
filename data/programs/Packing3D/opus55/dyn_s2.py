from env_client import make_env
env = make_env()
tf = env.observation_space.type_features
for seed in [0,1]:
    obs, info = env.reset(seed=seed)
    for name in obs.get_object_names():
        o = obs.get_object_from_name(name) if isinstance(name,str) else name
        if 'robot' in str(o): continue
        print(seed, o, {k:round(float(obs.get(o,k)),3) for k in tf[o.type]})
# which part is triangle
import collections; c=collections.Counter()
for seed in range(200):
    obs,info=env.reset(seed=seed)
    for name in obs.get_object_names():
        o = obs.get_object_from_name(name) if isinstance(name,str) else name
        c[(str(o),str(o.type))]+=1
print(c)

from env_client import make_env

env=make_env()
for seed in range(3):
 s,i=env.reset(seed=seed)
 print('SEED',seed, 'INFO',i)
 for name in sorted(s.get_object_names()):
  obj=s.get_object_from_name(name)
  print(name, {f:s.get(obj,f) for f in env.observation_space.type_features[obj.type]})
env.close()

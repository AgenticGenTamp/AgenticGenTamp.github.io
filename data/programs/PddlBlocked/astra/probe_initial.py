from env_client import make_env
for seed in [0,1,2,3,42]:
 e=make_env(); s,i=e.reset(seed=seed)
 print('SEED',seed,'INFO',i,flush=True)
 for t in e.observation_space.types:
  for o in s.get_objects(t):
   print(o.name, {f:round(s.get(o,f),5) for f in e.observation_space.type_features[t]},flush=True)
 e.close()

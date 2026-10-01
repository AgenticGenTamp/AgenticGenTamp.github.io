from env_client import make_env
E=make_env()
for seed in range(3):
 s,i=E.reset(seed=seed)
 print('SEED',seed,'INFO',i,'MAX',E.max_steps)
 for typ in E.observation_space.types:
  oo=s.get_objects(typ)
  if oo:
   print('TYPE',typ)
   for o in oo:
    print(o, {f:round(s.get(o,f),4) for f in E.observation_space.type_features[typ]})
E.close()

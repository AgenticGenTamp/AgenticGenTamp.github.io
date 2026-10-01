from env_client import make_env
E=make_env()
s,i=E.reset(seed=0)
for t in ('crv_robot','shelf','target_block'):
 typ=E.observation_space.get_type(t)
 for o in s.get_objects(typ):
  print(o, {f:s.get(o,f) for f in E.observation_space.type_features[typ]})
print('info',i,'maxsteps',E.max_steps)
E.close()

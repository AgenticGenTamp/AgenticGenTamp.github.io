from env_client import make_env
for n in [2,3,4]:
 e=make_env();s,info=e.reset(seed=0,options={'object_count':n});print('COUNT',n,info,flush=True)
 for name in sorted(s.get_object_names()):
  if not name.startswith('part'):continue
  o=s.get_object_from_name(name);print(name,o.type.name, {f:s.get(o,f) for f in ['triangle_type','side_a','side_b'] if f in e.observation_space.type_features[o.type]},flush=True)
 e.close()

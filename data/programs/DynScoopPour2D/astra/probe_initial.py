from env_client import make_env
E=make_env()
s,i=E.reset(seed=42)
print('info',i,'max',E.max_steps)
for name in sorted(s.get_object_names()):
 o=s.get_object_from_name(name)
 print(name,o.type.name,{f:round(s.get(o,f),4) for f in E.observation_space.type_features[o.type]})
E.close()

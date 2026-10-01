from env_client import make_env
E=make_env();s,i=E.reset(seed=0)
print('info',i)
for t in E.observation_space.types:
 print(t,[(o.name,[round(s.get(o,f),4) for f in E.observation_space.type_features[t]]) for o in s.get_objects(t)])
print('max',E.max_steps)
E.close()

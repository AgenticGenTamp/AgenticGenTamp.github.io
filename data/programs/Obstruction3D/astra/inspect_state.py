from env_client import make_env
E=make_env()
s,info=E.reset(seed=0)
print('INFO',info)
for typ in E.observation_space.types:
    for obj in s.get_objects(typ):
        print(obj.name, {f:s.get(obj,f) for f in E.observation_space.type_features[typ]})
E.close()

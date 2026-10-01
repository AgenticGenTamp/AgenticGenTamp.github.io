from env_client import make_env
env=make_env(); obs,_=env.reset(seed=0)
n=obs.get_object_names(); print(type(n), list(n)[:3], type(list(n)[0]))
o=obs.get_object_from_name('obstruction0'); print(type(o), o.name, o.type, type(o.type))
print(env.observation_space.get_type('dyn_rectangle'))
print([x.name for x in obs.get_objects(env.observation_space.get_type('dyn_rectangle'))])

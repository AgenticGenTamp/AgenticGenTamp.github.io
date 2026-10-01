from env_client import make_env
for seed in range(8):
 e=make_env();s,info=e.reset(seed=seed)
 print('SEED',seed,info)
 for typ in ['mujoco_fixture','mujoco_movable_object']:
  print(typ,[(o.name,[round(s.get(o,f),3) for f in ['x','y','z']]) for o in s.get_objects(e.observation_space.get_type(typ))])
 e.close()

from env_client import make_env
E=make_env()
for seed in range(100,400):
 s,info=E.reset(seed=seed);sh=s.get_objects(E.observation_space.get_type('shelf'))[0];x=s.get(sh,'x1');w=s.get(sh,'width1')
 if x<.08 or x+w>4.92:print(seed,info['object_count'],round(x,3),round(w,3),flush=True)
E.close()

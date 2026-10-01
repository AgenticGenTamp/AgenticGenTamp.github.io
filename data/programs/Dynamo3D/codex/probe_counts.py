from env_client import make_env

for seed in range(50):
    env=make_env(); s,info=env.reset(seed=seed)
    n=info['object_count']
    if n==12:
        rob=s.get_object_from_name('robot')
        chairs=s.get_objects(env.observation_space.get_type('mujoco_movable_object'))
        pts=[(float(s.get(o,'x')),float(s.get(o,'y'))) for o in chairs]
        print(seed,n,'robot',round(float(s.get(rob,'pos_base_x')),2),round(float(s.get(rob,'pos_base_y')),2),
              'chairs',round(min(x for x,y in pts),2),round(max(x for x,y in pts),2),round(min(y for x,y in pts),2),round(max(y for x,y in pts),2))
    env.close()

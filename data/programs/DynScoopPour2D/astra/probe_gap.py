from env_client import make_env
E=make_env();s,_=E.reset(seed=42);r=s.get_objects(E.observation_space.get_type('kin_robot'))[0]
for i in range(20):s,*_=E.step([0,0,0,0,.014])
print('gap',s.get(r,'finger_gap'),'robot',s.get(r,'x'),s.get(r,'y'))
E.close()

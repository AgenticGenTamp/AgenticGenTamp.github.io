from env_client import make_env
for count in [4,5]:
 e=make_env();s,i=e.reset(seed=0,options={'object_count':count});f=e.observation_space.get_type('mujoco_fixture');m=e.observation_space.get_type('mujoco_movable_object')
 print('META',count,'fixtcount',len(s.get_objects(f)),'fixtures',[(o.name,[round(s.get(o,v),4) for v in ['x','y','z']]) for o in s.get_objects(f)],'objects',[(o.name,[round(s.get(o,v),4) for v in ['bb_x','bb_y','bb_z']]) for o in s.get_objects(m)],flush=True);e.close()

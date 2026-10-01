import sys; sys.path.insert(0,'.')
from env_client import make_env
import numpy as np
env=make_env()
for seed,k in [(0,None),(1,None),(0,5),(3,None)]:
    obs,info=env.reset(seed=seed, options={} if k is None else {'object_count':k})
    F=env.observation_space.get_type('mujoco_fixture'); M=env.observation_space.get_type('mujoco_movable_object')
    print('seed',seed,k,'info',info)
    for o in obs.get_objects(F): print(' ',o.name,[round(obs.get(o,f),3) for f in ['x','y','z','qw','qz']])
    for o in obs.get_objects(M): print(' ',o.name,[round(obs.get(o,f),3) for f in ['x','y','z','qw','qz','bb_x','bb_y','bb_z']])
    print(' names',obs.get_object_names())
env.close()

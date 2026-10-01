from env_client import make_env
import numpy as np
env=make_env()
for s in range(20):
    obs,info=env.reset(seed=s)
    names=sorted(obs.get_object_names())
    cb=obs.get_object_from_name('cupboard_1')
    cp=[round(obs.get(cb,f),3) for f in ['x','y','z','qw','qx','qy','qz']]
    cs=[n for n in names if n.startswith('cube')]
    other=[n for n in names if not n.startswith('cube') and n not in('robot','cupboard_1')]
    c0=obs.get_object_from_name(cs[0]); 
    print(s,info,len(cs),cp,other, [round(obs.get(c0,f),3) for f in ['x','y','z','bb_x','bb_y','bb_z']])

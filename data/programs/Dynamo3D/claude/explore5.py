from env_client import make_env
import numpy as np
for s in [1,2,3,7,11]:
    env=make_env(); obs,info=env.reset(seed=s)
    names=sorted(obs.get_object_names())
    print(s, info, names)
    for n in names:
        if n=='robot': continue
        o=obs.get_object_from_name(n)
        print("   ",n,o.type.name,[round(float(obs.get(o,f)),3) for f in ['x','y','z','bb_x','bb_y','bb_z']])
    r=obs.get_object_from_name('robot')
    print("   robot",[round(float(obs.get(r,f)),3) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
    env.close()

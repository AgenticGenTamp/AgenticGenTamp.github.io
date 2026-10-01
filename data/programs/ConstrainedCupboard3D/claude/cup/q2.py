import sys; sys.path.insert(0,"/sandbox")
import numpy as np
from env_client import make_env
env=make_env()
for s in [0,1,7]:
    obs,info=env.reset(seed=s)
    print('=== seed',s)
    for n in sorted(obs.get_object_names()):
        o=obs.get_object_from_name(n); f=obs.type_features[o.type]
        if n.startswith('cuboid') or n=='robot':
            d={k:round(float(v),4) for k,v in zip(f,obs.data[o])}
            print('  ',n,{k:v for k,v in list(d.items())[:8]})
    print('   type of cupboard_0:',obs.get_object_from_name('cupboard_0').type)
    print('   feats:',obs.type_features[obs.get_object_from_name('cupboard_0').type])
env.close()

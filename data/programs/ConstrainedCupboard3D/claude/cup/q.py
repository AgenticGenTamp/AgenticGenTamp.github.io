import sys; sys.path.insert(0,"/sandbox")
import numpy as np, json
from env_client import make_env
env=make_env()
for s in [0,1,7]:
    obs,info=env.reset(seed=s)
    names=sorted(obs.get_object_names())
    print('=== seed',s,'names',names)
    for n in names:
        if n.startswith('cupboard'):
            o=obs.get_object_from_name(n); f=obs.type_features[o.type]
            print('  ',n,{k:round(float(v),4) for k,v in zip(f,obs.data[o])})
env.close()

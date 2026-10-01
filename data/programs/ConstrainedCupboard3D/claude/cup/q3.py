import sys; sys.path.insert(0,"/sandbox")
import numpy as np
from env_client import make_env
env=make_env()
for n in [2,3,4,5,6]:
    for s in [0,1,2,7]:
        obs,info=env.reset(seed=s, options={'object_count':n})
        nm=obs.get_object_names()
        cub=[c for c in nm if c.startswith('cuboid')]; cup=[c for c in nm if c.startswith('cupboard')]
        ys=sorted(round(float(obs.data[obs.get_object_from_name(c)][1]),3) for c in cup)
        print('oc=%d seed=%d ncub=%d ncup=%d ys=%s'%(n,s,len(cub),len(cup),ys))
env.close()

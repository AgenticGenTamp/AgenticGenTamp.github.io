import numpy as np
from env_client import make_env
from probe_term_lib import *
for s in [1,2,3]:
    env=make_env(); obs,info=env.reset(seed=s,options={'object_count':12})
    print('seed',s,{n:np.round(cxy(obs,n),2).tolist() for n in chairs(obs)},'rob',np.round(rxy(obs),2).tolist(),flush=True)
    env.close()

import numpy as np
from env_client import make_env
from probe_term_lib import *
for c in [0,1,2,3,4,5,6]:
    for s in [1,2]:
        try:
            env=make_env(); obs,info=env.reset(seed=s,options={'object_count':c})
            print('count',c,'seed',s,info.get('object_count'),{n:np.round(cxy(obs,n),2).tolist() for n in chairs(obs)},'rob',np.round(rxy(obs),2).tolist(),flush=True)
            env.close()
        except Exception as e: print('count',c,'seed',s,'ERR',repr(e)[:120],flush=True)

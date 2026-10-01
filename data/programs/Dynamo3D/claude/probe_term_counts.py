import numpy as np
from env_client import make_env
from probe_term_lib import *
for s in range(0,16):
    env=make_env(); obs,info=env.reset(seed=s)
    print(s, info.get('object_count'), {n:np.round(cxy(obs,n),2).tolist() for n in chairs(obs)}, 'rob',np.round(rxy(obs),2).tolist(),flush=True)
    env.close()

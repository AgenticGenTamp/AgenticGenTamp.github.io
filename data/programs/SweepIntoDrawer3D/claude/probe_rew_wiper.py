import numpy as np
from env_client import make_env
for s in range(20,30):
    env=make_env(); obs,_=env.reset(seed=s)
    print(s, np.round(obs[147:150],3).tolist(), "quat",np.round(obs[150:154],2).tolist(),"bb",np.round(obs[160:163],3).tolist())
    env.close()

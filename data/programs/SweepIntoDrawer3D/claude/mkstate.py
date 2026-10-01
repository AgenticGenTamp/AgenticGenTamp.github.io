import numpy as np, json, sys
from env_client import make_env
env=make_env(); obs,_=env.reset(seed=0); env.close()
np.save('obs0.npy',obs)
print(list(np.round(obs,4)))

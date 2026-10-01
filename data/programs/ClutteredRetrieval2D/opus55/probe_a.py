import numpy as np
from env_client import make_env
env=make_env()
obs,info=env.reset(seed=0)
def dump(obs):
    for o in sorted(obs.data,key=lambda o:o.name):
        print(o.name,{f:round(float(obs.get(o,f)),4) for f in obs.type_features[o.type]})
dump(obs); print(info)

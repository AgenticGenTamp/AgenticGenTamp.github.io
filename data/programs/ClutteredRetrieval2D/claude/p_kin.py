import numpy as np, math
from env_client import make_env
from util import dump, g
env = make_env()
obs,_ = env.reset(seed=0)
st = env.get_state()
print("state type", type(st))
names = sorted(obs.get_object_names())
print("objects:", names)
d = dump(obs)
for n in names: print(n, d[n])

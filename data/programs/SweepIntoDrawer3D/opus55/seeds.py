import numpy as np
from env_client import make_env
np.set_printoptions(precision=3, suppress=True, linewidth=150)
env=make_env()
for s in range(12):
    o,_=env.reset(seed=s)
    C=o[:80].reshape(5,16)[:,:3]
    print(s, "cubes x[%.2f,%.2f] y[%.2f,%.2f]"%(C[:,0].min(),C[:,0].max(),C[:,1].min(),C[:,1].max()), "robot",o[125:128],"wiper",o[147:150], o[150:154], "isl",o[96:99])

import numpy as np
from env_client import make_env
env=make_env()
# probe boundaries in each direction from origin
for dirn in [(1,0),(-1,0),(0,1),(0,-1)]:
    o,i=env.reset(seed=72)
    a=np.zeros(11); a[:2]=np.array(dirn)*0.4
    for t in range(15): o,*_=env.step(a)
    print(dirn,o[:2])

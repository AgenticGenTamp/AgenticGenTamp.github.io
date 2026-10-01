import numpy as np
from env_client import make_env
from kin import fk_world
env=make_env(); o,_=env.reset(seed=0)
print(o[93:96])
for t in range(24):
    a=np.zeros(11,np.float32); a[2]=0.1 if t<12 else -0.1
    o,*_=env.step(a)
    if t%3==2: print(t,o[93:96].round(4), fk_world(o[93:96],o[96:103])[0].round(3))

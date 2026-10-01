import numpy as np
from env_client import make_env
env=make_env()
np.set_printoptions(precision=3,suppress=True)
for s in range(5):
    o,i=env.reset(seed=s); print(s,o, i)
o,i=env.reset(seed=0)
a=np.zeros(11); a[0]=0.4
for t in range(3):
    o,r,te,tr,i=env.step(a); print(o[:3],r,te,tr)
a=np.zeros(11); a[2]=0.4
o,r,te,tr,i=env.step(a); print(o[:3],r,te,tr)

from env_client import make_env
import numpy as np
from envutil import Sim
env=make_env(); S=Sim(env,env.reset(seed=1)[0])
S.moveto([3.5,-1.3,0],S.q())
for k in range(30):
    b=S.base(); S.step(np.r_[0.1,0,0,np.zeros(8)])
    if np.abs(S.base()-b).max()<1e-6: print('blocked at',b.round(3)); break
print(S.base().round(3))
for dy in [-0.1,0.1]:
    b=S.base(); S.step(np.r_[0,dy,0,np.zeros(8)]); print('dy',dy,S.base().round(3))

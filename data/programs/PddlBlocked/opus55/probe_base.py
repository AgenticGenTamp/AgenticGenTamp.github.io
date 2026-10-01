from env_client import make_env
import numpy as np
from envutil import Sim
env = make_env()
S=Sim(env,env.reset(seed=0)[0])
for i in range(40):
    x0=S.base()[0]; S.step(np.r_[0.02*5,0,0,np.zeros(8)])
    if abs(S.base()[0]-x0)<1e-6: break
for i in range(20):
    x0=S.base()[0]; S.step(np.r_[0.01,0,0,np.zeros(8)])
    if abs(S.base()[0]-x0)<1e-6: break
print('blocked at yaw0 base', S.base())
# now arm lowered: tuck arm low?  try rotating
for i in range(5):
    b=S.base(); S.step(np.r_[0,0,0.2,np.zeros(8)]); print('rot', S.base())

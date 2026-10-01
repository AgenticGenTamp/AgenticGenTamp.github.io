from env_client import make_env
import numpy as np
from envutil import Sim
env=make_env(); S=Sim(env,env.reset(seed=0)[0])
for k in range(15): S.step(np.r_[0,0.2,np.zeros(9)])
print('y after +3',S.base())
for k in range(20): S.step(np.r_[0.2,0,np.zeros(9)])
print('x after +4',S.base())
for k in range(30): S.step(np.r_[0,-0.2,np.zeros(9)])
print('y after -6',S.base())

from env_client import make_env
import numpy as np, kin
from rutil import Bot
env=make_env(); obs,_=env.reset(seed=0); b=Bot(env,obs)
q0=b.q(); qt=q0+np.array([0.3,0.3,0.2,0.3,0.2,0.3,0.2])
for k in range(60):
    b.act(q_t=qt)
    if k%4==0: print(k,np.round(qt-b.q(),4))

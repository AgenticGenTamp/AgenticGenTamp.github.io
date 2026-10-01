import numpy as np
from env_client import make_env
from envutil import Sim
env=make_env(); S=Sim(env,env.reset(seed=74)[0])
print('start q', np.round(S.q(),2), 'base', np.round(S.base(),2))
S.moveto(np.array([1.0,0,-3.13]),S.q())
c=[2.04,0.25,-0.65,-0.2,-0.67,-0.34,1.29]
ok=S.moveto(S.base(),np.array(c)); print('free space reached', ok, np.round(S.q(),2))

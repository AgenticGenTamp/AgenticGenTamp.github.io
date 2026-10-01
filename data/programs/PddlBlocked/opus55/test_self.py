import numpy as np
from env_client import make_env
from envutil import Sim
env=make_env(); S=Sim(env,env.reset(seed=0)[0])
S.moveto(np.array([1.0,0,0]),S.q())
cfgs=[[1.98,0.24,0.23,-0.42,-1.69,-1.87,1.53],[2.1,0.22,1.43,-1.6,-2.87,-1.53,1.71],[1.91,0.21,1.56,-0.2,-2.81,-0.64,1.22],
      [2.14,0.15,1.93,-1.79,-1.98,-0.38,0.6],[1.54,0.23,1.72,-1.38,-2.89,-1.32,1.38],[1.89,0.29,1.23,-1.8,-2.98,-1.66,2.0],[1.68,0.24,1.75,-1.47,-2.86,-1.28,1.37]]
for c in cfgs:
    ok=S.moveto(S.base(),np.array(c))
    print(c, 'reached', ok, np.round(S.q(),2))
    S.moveto(S.base(),np.array([0.39,0.33,0.0,-1.52,2.72,-1.22,-2.99]))

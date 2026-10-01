import numpy as np
from env_client import make_env
env=make_env(); obs,_=env.reset(seed=0)
print(env.action_space.low, env.action_space.high, env.action_space.dtype)
for a in [np.zeros(5), np.array([0,0,0.09817477042468103,0,0]), np.array([0,0,-0.09817477042468103,0,0]), np.array([0.03,0,0,0,0])]:
    try: env.step(a); print('ok',a)
    except Exception as e: print('fail',a,e)

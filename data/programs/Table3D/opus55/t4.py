import time, numpy as np
from env_client import make_env
from approach import *
env=make_env(); obs,info=env.reset(seed=0)
ap=GeneratedApproach(env.action_space, env.observation_space, {})
x0=ap._x(obs); c=obs.get_object_from_name('cube0')
p,cy=ap._cube_info(obs,c)
for k in range(4):
    t=time.time(); sol=solve_min_step(x0,p,cy+k*np.pi/2); print(k, None if sol is None else round(sol[1],3), round(time.time()-t,2), flush=True)
env.close()

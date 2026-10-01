import numpy as np
from env_client import make_env
from approach import *
from relaxed import solve_relaxed, world_T
env=make_env(); ap=GeneratedApproach(env.action_space, env.observation_space, {})
obs,info=env.reset(seed=0); x0=ap._x(obs)
np.set_printoptions(precision=3,suppress=True)
for n,o in ap._cubes(obs)[:2]:
    p,cy=ap._cube_info(obs,o)
    for tm in [0.5,1.0,1.5]:
        sol=solve_relaxed(x0,p,cy+np.pi/2,tilt_max=tm)
        if sol:
            print(n,tm,round(sol[1],3),sol[0]-x0, 'Rz',world_T(sol[0])[1][:,2])
        sol=solve_relaxed(x0,p,cy+np.pi/2,tilt_max=tm,xi=x0.copy())
        if sol: print('  from x0',round(sol[1],3))
env.close()

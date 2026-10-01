import numpy as np, sys
from env_client import make_env
from approach import *
env=make_env(); seed=int(sys.argv[1]); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {})
x0=ap._x(obs); cubes=ap._cubes(obs)
for n,o in cubes:
    p,cy=ap._cube_info(obs,o)
    for k in range(4):
        cl=ap._finger_clear(obs,n,p,cy+k*np.pi/2,cubes)
        sol=solve_min_step(x0,p,cy+k*np.pi/2)
        print(n,k,round(cl,3), None if sol is None else round(sol[1],3))
env.close()

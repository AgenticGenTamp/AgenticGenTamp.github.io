import numpy as np, sys
from env_client import make_env
from approach import *
env=make_env(); seed=int(sys.argv[1]); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
name,acts,exp=ap.fast
for n,o in ap._cubes(obs): print(n, np.round(ap._cube_info(obs,o)[0],3), ap._cube_info(obs,o)[1])
print('target',name)
for i in range(len(acts)-1):
    obs,*_=env.step(acts[i]); print('dev',np.abs(ap._x(obs)-exp[i]).max())
x=ap._x(obs); p,R=world_flange(x); print('flange world',p.round(3),'R',R.round(2), 'base', x[:3].round(3))
env.close()

import numpy as np
from env_client import make_env
from probe_lib import *
env=make_env()
obs,_=env.reset(seed=275)
obs,_=goto(env,obs,[None,1.4,None,None,None],maxsteps=100)
obs=seq(env,obs,[1.6,1.4,-np.pi/2,0.24,0.32],order=(0,2))
for L in [0.24,0.48]:
    obs=seq(env,obs,[1.6,1.4,None,L,0.32],order=(0,1,3,4))
    obs=move_until_blocked(env,obs,0,-1); print('down L',L,'xmin',rob(obs).round(4))
    obs=move_until_blocked(env,obs,0,+1); print('down L',L,'xmax',rob(obs).round(4))
    obs=seq(env,obs,[1.6,1.2,None,None,None],order=(0,1))
    obs=move_until_blocked(env,obs,1,-1); print('down L',L,'ymin',rob(obs).round(4))
obs=seq(env,obs,[1.6,1.4,np.pi,0.24,0.32],order=(1,0,2))
for L in [0.24,0.48]:
  for g in [0.12,0.32]:
    obs=seq(env,obs,[1.6,1.4,None,L,g],order=(0,1,3,4))
    obs=move_until_blocked(env,obs,0,-1); print('th=pi L',L,'g',g,'xmin',rob(obs).round(4))

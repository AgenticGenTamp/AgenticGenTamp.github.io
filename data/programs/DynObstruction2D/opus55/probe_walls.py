import numpy as np
from env_client import make_env
from util import *
env=make_env()
obs,_=env.reset(seed=275)
print(rob(obs).round(3))
obs,_=goto(env,obs,[None,1.4,-np.pi/2,0.24,0.32],maxsteps=100)
print('up',rob(obs).round(3))
obs,_=goto(env,obs,[None,2.5,None,None,None],maxsteps=60); print('ymax',rob(obs).round(3))
obs,_=goto(env,obs,[None,1.4,None,None,None],maxsteps=60)
for arm in [0.24,0.48]:
    obs,_=goto(env,obs,[None,None,None,arm,None])
    obs,_=goto(env,obs,[-1,None,None,None,None],maxsteps=150); print('arm',arm,'xmin',rob(obs).round(4))
    obs,_=goto(env,obs,[5,None,None,None,None],maxsteps=150); print('arm',arm,'xmax',rob(obs).round(4))
# min y with arm down at arm .48
obs,_=goto(env,obs,[1.6,None,None,None,None],maxsteps=150)
obs,_=goto(env,obs,[None,0.0,None,None,None],maxsteps=150); print('ymin arm.48',rob(obs).round(4))
obs,_=goto(env,obs,[None,None,0.0,None,None],maxsteps=150); 
obs,_=goto(env,obs,[None,0.0,None,None,None],maxsteps=150); print('ymin theta0',rob(obs).round(4))

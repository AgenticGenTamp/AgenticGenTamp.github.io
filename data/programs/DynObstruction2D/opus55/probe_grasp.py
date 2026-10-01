import numpy as np, sys
from env_client import make_env
from probe_lib import *
env=make_env()
def blk(obs):
    B=obs.get_object_from_name('target_block')
    return {f:obs.get(B,f) for f in ['x','y','theta','width','height','held']}
for s in [283,70,54,97,170,189,217,219,157,298]:
    obs,_=env.reset(seed=s); b=blk(obs)
    obs,_=goto(env,obs,[None,1.4,None,None,None],maxsteps=100)
    obs=seq(env,obs,[b['x'],1.4,-np.pi/2,0.48,0.32],order=(2,3,4,0))
    top=0.1+b['height']
    # descend to fingertips 0.02 above floor
    obs,_=goto(env,obs,[None,0.1+0.68+0.02,None,None,None],maxsteps=150)
    obs,_=goto(env,obs,[None,None,None,None,0.12],maxsteps=30)
    b2=blk(obs)
    obs,_=goto(env,obs,[None,1.0,None,None,None],maxsteps=30)
    b3=blk(obs)
    print(s,'bx %.3f w %.3f'%(b['x'],b['width']),'rob',rob(obs).round(3),'held',b2['held'],'after lift y %.3f th %.2f'%(b3['y'],b3['theta']))

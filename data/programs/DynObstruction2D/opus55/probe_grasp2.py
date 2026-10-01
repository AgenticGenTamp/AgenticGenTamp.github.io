import numpy as np, sys
from env_client import make_env
from probe_lib import *
env=make_env()
def blk(obs):
    B=obs.get_object_from_name('target_block')
    return {f:round(obs.get(B,f),3) for f in ['x','y','theta','width','height','held']}
s=int(sys.argv[1]); depth=float(sys.argv[2])
obs,_=env.reset(seed=s); b=blk(obs); print(b)
obs=prep(env,obs)
obs=seq(env,obs,[b['x'],None,None,0.48,0.32],order=(3,4,0))
top=0.1+b['height']; tip=max(0.12,top-depth)
obs,_=goto(env,obs,[None,tip+0.68+0.03,None,None,None],maxsteps=150); print(rob(obs).round(3),blk(obs))
for i in range(6):
    obs,_,_,_,_=step(env,[0,-0.005,0,0,0])
print(rob(obs).round(3),blk(obs))
for i in range(12):
    obs,_,_,_,_=step(env,[0,0,0,0,-0.0199]); 
    if i%3==2: print(rob(obs).round(3),blk(obs))

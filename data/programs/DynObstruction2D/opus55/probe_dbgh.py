import numpy as np
import probe_hybrid as H
from probe_lib import *
env=H.env
obs,_=env.reset(seed=14); b=H.blk(obs); print(b)
obs=prep(env,obs); print('prep',rob(obs).round(3))
obs=seq(env,obs,[None,None,np.pi/2,None,None],order=(2,)); print('rot',rob(obs).round(3))
obs=seq(env,obs,[H.W-0.2401,None,None,None,None],order=(0,)); print('x',rob(obs).round(3))
for i in range(400):
    r0=rob(obs); obs,_,_,_,_=step(env,[0,-0.005,0,0,0]); bb=H.blk(obs)
    if H.gapof(bb,'R')>=0.45 or rob(obs)[1]>r0[1]-0.0025: break
print('desc',rob(obs).round(3),H.gapof(bb,'R'))
obs,_=goto(env,obs,[None,1.2,None,None,None],maxsteps=60); print('up',rob(obs).round(3))
obs=seq(env,obs,[None,None,-np.pi/2,0.48,0.32],order=(2,3,4)); print('armdown',rob(obs).round(3))
obs,_=goto(env,obs,[None,0.8,None,None,None],maxsteps=60); print('low',rob(obs).round(3), H.gapof(H.blk(obs),'R'))
for i in range(10):
    obs,_,_,_,_=step(env,[-0.02,0,0,0,0]); print(rob(obs).round(3), H.gapof(H.blk(obs),'R'))

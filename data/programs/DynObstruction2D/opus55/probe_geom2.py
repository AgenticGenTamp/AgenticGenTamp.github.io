import numpy as np
from env_client import make_env
from probe_lib import *
from probe_solver import rects
env=make_env()
obs,_=env.reset(seed=275)
obs,_=goto(env,obs,[None,1.4,None,None,None],maxsteps=100)
obs=seq(env,obs,[1.6,1.4,-np.pi/2,0.48,0.32],order=(0,2,3))
for th in [-np.pi+0.5,-np.pi/2-0.9]:
  for g in [0.12,0.32]:
    obs=seq(env,obs,[1.6,1.4,th,0.48,g],order=(1,0,2,3,4))
    obs=move_until_blocked(env,obs,1,-1); r=rob(obs)
    V=np.vstack(rects(*r)); V2=np.vstack(rects(r[0],r[1],r[2],r[3],r[4]+0.06))
    print('th %.3f g %.2f ymin %.4f  model lowest %.4f  model(inner-gap) %.4f'%(th,g,r[1],V[:,1].min(),V2[:,1].min()))
    obs=seq(env,obs,[1.6,1.4,th,0.48,g],order=(1,))
    obs=move_until_blocked(env,obs,0,-1); r=rob(obs)
    V=np.vstack(rects(*r)); V2=np.vstack(rects(r[0],r[1],r[2],r[3],r[4]+0.06))
    print('   xmin %.4f  model leftmost %.4f  model(inner-gap) %.4f'%(r[0],V[:,0].min(),V2[:,0].min()))
    obs=seq(env,obs,[1.6,1.4,th,0.48,g],order=(0,))

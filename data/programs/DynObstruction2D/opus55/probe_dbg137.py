import numpy as np, sys
import probe_hybrid as H
from probe_lib import *
env=H.env; s=int(sys.argv[1]); xo=float(sys.argv[2]) if len(sys.argv)>2 else 0.2401
obs,_=env.reset(seed=s); obs=prep(env,obs); obs=rotate_by(env,obs,np.pi)
obs=seq(env,obs,[H.W-xo,None,None,None,None],order=(0,))
for i in range(400):
    r0=rob(obs); obs,_,_,_,_=step(env,[0,-0.005,0,0,0]); b=H.blk(obs)
    if i%4==0 or abs(b['theta'])>0.05: print(i,rob(obs).round(3)[:2],'bx %.3f by %.3f th %.3f gap %.3f'%(b['x'],b['y'],b['theta'],H.gapof(b,'R')))
    if H.gapof(b,'R')>=0.45 or rob(obs)[1]>r0[1]-0.0025: break
for i in range(10):
    obs,_,_,_,_=step(env,[0,0.045,0,0,0]); b=H.blk(obs); print('up',rob(obs).round(3)[:2],'th %.3f'%b['theta'])

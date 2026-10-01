import numpy as np, sys
import probe_hybrid as H
from probe_lib import *
env=H.env; s=int(sys.argv[1]); dy=float(sys.argv[2]); xo=float(sys.argv[3]) if len(sys.argv)>3 else 0.2401
obs,_=env.reset(seed=s); b=H.blk(obs); top=0.1+b['height']
obs=prep(env,obs); obs=rotate_by(env,obs,np.pi)
obs=seq(env,obs,[H.W-xo,top+0.28,None,None,None],order=(0,1))
mx=0
for i in range(2000):
    r0=rob(obs); obs,_,_,_,_=step(env,[0,-dy,0,0,0]); b=H.blk(obs); mx=max(mx,abs(b['theta']))
    if H.gapof(b,'R')>=0.45 or rob(obs)[1]>r0[1]-dy/2: break
print(s,dy,xo,'end',rob(obs).round(3)[:2],'th %.3f maxth %.3f gap %.3f'%(b['theta'],mx,H.gapof(b,'R')))

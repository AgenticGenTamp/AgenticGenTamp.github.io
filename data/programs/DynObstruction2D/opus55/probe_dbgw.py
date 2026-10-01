import numpy as np, sys
from env_client import make_env
from probe_lib import *
from probe_solver import rects
env=make_env()
seed=int(sys.argv[1]); a=float(sys.argv[2]); g=float(sys.argv[3]); off=float(sys.argv[4])
obs,_=env.reset(seed=seed)
B=obs.get_object_from_name('target_block')
bx=obs.get(B,'x'); w=obs.get(B,'width'); h=obs.get(B,'height'); e=bx-w/2; top=0.1+h
th=-np.pi/2-a; L=0.36
R=rects(0,0,th,L,g); V=R[1]; iv=V[:,1].argmin(); cx,cy=V[iv]
xr=e-off-cx; yr=top+0.01-cy
obs=prep(env,obs)
obs=seq(env,obs,[None,None,th,L,g],order=(2,3,4))
obs=seq(env,obs,[xr,1.5,None,None,None],order=(1,0))
obs=seq(env,obs,[xr,yr+0.05,th,L,g],order=(1,))
print('start',rob(obs).round(3),'e',round(e,3),'top',round(top,3))
for i in range(60):
    obs,_,_,_,_=step(env,[0,-0.005,0,0,0])
    r=rob(obs); VV=np.vstack(rects(*r))
    print(i,'y %.4f lowest %.3f,%.3f'%(r[1],*VV[VV[:,1].argmin()]),'blk x %.4f y %.4f th %.3f'%(obs.get(B,'x')-w/2,obs.get(B,'y'),obs.get(B,'theta')))

import numpy as np
from probe_vis_lib import *
env=make_env(); obs,info=env.reset(seed=39, options={'object_count':1})
L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
print("O",O)
free,xs,res=build_grid(obs)
def vis(obs,i):
    if rf(obs,i)['calibrated']>0.5: obs,_,_,_,_=st(env,op='image',i=i)
    obs,_,_,_,_=st(env,op='calibrate',i=i)
    return obs, rf(obs,i)['calibrated']>0.5
tests=[(1,(-0.40,1.20),'west of wall, cross wall'),
       (1,(-0.40,0.60),'west, farther'),
       (1,(-1.20,1.60),'west, cross wall, oblique'),
       (0,( 1.40,1.20),'east control same dist'),
       (0,( 0.40,-0.10),'east straight below, d~2.16 (out of range ctrl)')]
for i,P,lbl in tests:
    obs,ok=nav(env,obs,P[0],P[1],i=i,tol=0.01,free=free,xs=xs,res=res)
    p=pose(obs,i)[:2]; d=np.linalg.norm(p-O)
    obs,v=vis(obs,i)
    print("%-42s rover%d pos=%s d=%.3f vis=%s reach=%s"%(lbl,i,np.round(p,3),d,v,ok))
env.close()

from probe_vis_lib import *
import numpy as np
env,obs,info=new_env(3)
L=layout(obs); Lp=np.array([-1.9,-2.0])
O=np.array([L['objective0']['x'],L['objective0']['y']])
obs,ok=nav(env,obs,O[0],O[1]-1.5,i=0)
obs,_,_,_,_=st(env,op='calibrate',i=0); obs,_,_,_,_=st(env,op='image',i=0)
obs,ok=nav(env,obs,1.06,-2.0,i=0); print("start",pose(obs,0))
for k in range(35):
    p=pose(obs,0)[:2]; d=np.linalg.norm(p-Lp)
    obs,_,_,_,_=st(env,op='send',i=0)
    r=feats(obs,'objective0')['received_image']
    print("x=%.3f y=%.3f d=%.4f recv=%.0f"%(p[0],p[1],d,r))
    if r>0.5: break
    obs,_=goto(env,obs,p[0]-0.02,-2.0,i=0,tol=0.003,maxit=6)
env.close()

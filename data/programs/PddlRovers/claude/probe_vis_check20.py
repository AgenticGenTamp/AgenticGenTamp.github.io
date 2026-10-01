import numpy as np
from probe_vis_lib import *
A=-0.085
for SEED,PIL in [(20,'obstacle7'),(23,'obstacle5')]:
    env=make_env(); obs,info=env.reset(seed=SEED, options={'object_count':1})
    L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
    q=np.array([L[PIL]['x'],L[PIL]['y']]); dqo=np.linalg.norm(q-O); u=(q-O)/dqo
    free,xs,res=build_grid(obs)
    e=np.array([np.cos(np.pi),np.sin(np.pi)])
    P=O+u*1.4-A*e
    obs,ok=nav(env,obs,P[0],P[1],i=0,tol=0.01,free=free,xs=xs,res=res)
    p=pose(obs,0); print("seed%d O=%s pillar=%s rover=%s theta=%.2f"%(SEED,np.round(O,3),np.round(q,3),np.round(p[:2],3),p[2]))
    obs,_,_,_,_=st(env,op='calibrate',i=0)
    print("   calibrated=%.0f"%rf(obs,0)['calibrated'])
    obs,_,_,_,_=st(env,op='image',i=0)
    print("   after image: calib=%.0f have_image_r0=%.0f"%(rf(obs,0)['calibrated'],feats(obs,'objective0')['have_image_rover0']))
    env.close()

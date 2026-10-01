import numpy as np
from probe_vis_lib import *
env=make_env(); obs,info=env.reset(seed=13, options={'object_count':1})
L=layout(obs); O=np.array([L['objective0']['x'],L['objective0']['y']])
free,xs,res=build_grid(obs)
P=O+np.array([-0.5,-0.9])
obs,ok=nav(env,obs,P[0],P[1],i=0,tol=0.01,free=free,xs=xs,res=res)
print("pos",np.round(pose(obs,0)[:2],3),"d",np.linalg.norm(pose(obs,0)[:2]-O),ok)
for k in range(4):
    obs,_,_,_,_=st(env,op='calibrate',i=0)
    c=rf(obs,0)['calibrated']
    obs,_,_,_,_=st(env,op='image',i=0)
    c2=rf(obs,0)['calibrated']; h=feats(obs,'objective0')['have_image_rover0']
    print("cycle%d after_calib=%s after_image calib=%s have=%s"%(k,c,c2,h))
env.close()

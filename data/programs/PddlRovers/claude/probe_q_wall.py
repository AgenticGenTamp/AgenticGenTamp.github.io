from probe_vis_lib import *
import numpy as np
env,obs,info=new_env(2)
L=layout(obs)
objs={n:np.array([L[n]['x'],L[n]['y']]) for n in L if n.startswith('objective')}
print({k:np.round(v,3).tolist() for k,v in objs.items()})
obs,ok=nav(env,obs,0.40,1.50,i=0); p=pose(obs,0)[:2]; print("nav",ok,p)
for k in range(4):
    p=pose(obs,0)[:2]
    print("dists:",{n:round(float(np.linalg.norm(p-v)),3) for n,v in objs.items()})
    obs,_,_,_,_=st(env,op='calibrate',i=0); c=rf(obs,0)['calibrated']
    obs,_,_,_,_=st(env,op='image',i=0)
    print(" calib=%.0f -> "%c, {n:(feats(obs,n)['have_image_rover0'],rf(obs,0)['calibrated']) for n in objs})
# now far from east objective, only west within 2m? move south-west-ish
obs,ok=nav(env,obs,0.30,0.20,i=0); p=pose(obs,0)[:2]
print("pos2",p,{n:round(float(np.linalg.norm(p-v)),3) for n,v in objs.items()})
env.close()

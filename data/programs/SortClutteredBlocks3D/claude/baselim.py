import numpy as np
from env_client import make_env
from arm import *
env=make_env(); obs,info=env.reset(seed=0)
b,q,g=robot_state(obs)
def drive(bt,n):
    global b,q,g,obs
    for t in range(n):
        a=make_action(b,q,bt,q,0.0); obs,rw,te,tr,i=env.step(a); b,q,g=robot_state(obs)
    return b.copy()
print('start',np.round(b,3))
print('->x0   ', np.round(drive(np.array([0.0,0.078,np.pi]),40),3))
print('->y+1  ', np.round(drive(np.array([0.0,1.0,np.pi]),40),3))
print('->x-1  ', np.round(drive(np.array([-1.0,1.0,np.pi]),40),3))
print('->y-1  ', np.round(drive(np.array([-1.0,-1.0,np.pi]),40),3))
print('->x0y0 ', np.round(drive(np.array([0.0,0.0,np.pi]),60),3))
print('cubes',{k:np.round(v,3) for k,v in cube_dict(obs).items()})
env.close()

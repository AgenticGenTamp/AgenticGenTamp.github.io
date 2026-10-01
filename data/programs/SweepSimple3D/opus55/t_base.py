from env_client import make_env
import numpy as np
from rutil import Bot
env=make_env(); obs,_=env.reset(seed=0); b=Bot(env,obs)
print('start',b.base())
for k in range(8): b.step([0,0,0.1,0,0,0,0,0,0,0,0])
print('after rot',np.round(b.base(),3))
for k in range(3):
    p=b.base(); b.step([0.1,0,0,0,0,0,0,0,0,0,0]); print('dx',np.round(b.base()-p,3))
for k in range(3):
    p=b.base(); b.step([0,0.1,0,0,0,0,0,0,0,0,0]); print('dy',np.round(b.base()-p,3))
ok=b.goto(base_t=[1.4,1.2,-np.pi/2]); print(ok,b.nsteps,np.round(b.base(),4))
for g in [1.0,0.0,0.5,1.0]:
    for k in range(6):
        b.act(grip=g); print(g,round(b.gripper(),3),round(b.obs.get(b.R,'vel_gripper'),3))

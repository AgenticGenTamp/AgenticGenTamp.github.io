from env_client import make_env
from ctl import act, rget, oget
import numpy as np
env=make_env()
for seed in [3,4,6,1,2]:
    obs,info=env.reset(seed=seed)
    hx,hy,hth=oget(obs,'hook','x'),oget(obs,'hook','y'),oget(obs,'hook','theta')
    u=np.array([np.cos(hth),np.sin(hth)]); nv=np.array([-np.sin(hth),np.cos(hth)])
    fe=np.array([hx,hy])-1.4*u-0.0533*nv
    base=fe-0.545*u
    print(seed,"hook",round(hx,3),round(hy,3),round(hth,3),"fe",np.round(fe,3),"base",np.round(base,3),
          "robot",round(rget(obs,'x'),3),round(rget(obs,'y'),3))
env.close()

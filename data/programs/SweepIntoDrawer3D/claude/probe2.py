import numpy as np
from env_client import make_env
np.set_printoptions(precision=4, suppress=True, linewidth=250)
env=make_env()
obs,info=env.reset(seed=0)
base=obs[125:136].copy()
# push action index 0 max
for idx in range(11):
    o=obs
    a=np.zeros(11); a[idx]=0.1
    env2=make_env(); o,_=env2.reset(seed=0)
    for k in range(5):
        o,r,t,tr,_=env2.step(a)
    print(idx, "robot", o[125:136]-obs[125:136], "r",r)
    env2.close()
env.close()

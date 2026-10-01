import numpy as np
from env_client import make_env
env=make_env()
np.set_printoptions(precision=3,suppress=True,linewidth=200)
for s in [0,1,2]:
    obs,info=env.reset(seed=s)
    print('seed',s,'info',info)
    for i in range(0,48,16): print(obs[i:i+16])
    print('furn',obs[48:93])
    print('robot',obs[93:115])
obs,r,te,tr,info=env.step(np.zeros(11,dtype=np.float32))
print(r,te,tr,info)
print('robot',obs[93:115])

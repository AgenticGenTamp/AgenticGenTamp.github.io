import numpy as np
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=0)
np.set_printoptions(precision=3,suppress=True,linewidth=150)
for i in range(5): print('cube',i,obs[16*i:16*i+16])
print('island',obs[96:103]); print('drawers',obs[103:109]); print('rest',obs[109:125])
print('base',obs[125:128]); print('q',obs[128:135]); print('grip',obs[135:147]); print('wiper',obs[147:163])
print(info)

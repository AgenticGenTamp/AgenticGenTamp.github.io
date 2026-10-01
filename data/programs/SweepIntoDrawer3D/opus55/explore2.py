import numpy as np
from env_client import make_env
np.set_printoptions(precision=4, suppress=True, linewidth=150)
env = make_env()
obs, info = env.reset(seed=0)
def pr(o): print(o[125:136])
pr(obs)
for j in [3,4,6]:
  for t in range(12):
    a=np.zeros(11,dtype=np.float32); 
    if t<4: a[j]=0.1
    obs,r,te,tr,info=env.step(a); pr(obs)
  print('---')
for t in range(6):
    a=np.zeros(11,dtype=np.float32); a[1]=0.1 if t<3 else 0
    obs,r,te,tr,info=env.step(a); pr(obs)
env.close()

from env_client import make_env
import numpy as np
np.set_printoptions(precision=4,suppress=True)
env=make_env()
for seed in [0,1,2,42]:
 s,info=env.reset(seed=seed)
 print('SEED',seed,'INFO',info,'MAX',env.max_steps)
 print('OBJECTS',s[:48].reshape(3,16))
 print('ROOM',s[48:93]);print('ROBOT',s[93:])
 for i in range(2):
  s,r,t,tr,info=env.step(np.zeros(11))
  print('ZERO',r,t,tr,info,'ROBOT',s[93:104])
env.close()

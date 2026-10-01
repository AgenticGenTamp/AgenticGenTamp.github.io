from env_client import make_env
import numpy as np,time
env=make_env(); obs,info=env.reset(seed=3)
t=time.time()
for i in range(100): obs,*_=env.step(np.zeros(11,np.float32))
print('step',(time.time()-t)/100)

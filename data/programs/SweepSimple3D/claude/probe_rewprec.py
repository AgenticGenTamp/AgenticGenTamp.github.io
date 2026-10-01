from env_client import make_env
import numpy as np
env=make_env()
obs,info=env.reset(seed=1)
for i in range(3):
    obs,rew,t,tr,info=env.step(np.zeros(11,dtype=np.float32))
    print(repr(rew))
env.close()

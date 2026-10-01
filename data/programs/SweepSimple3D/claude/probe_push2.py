from env_client import make_env
import numpy as np
env = make_env(); os_=env.observation_space; T=os_.get_type
obs,info = env.reset(seed=0)
print("n",info)
for i in range(5):
    a=np.zeros(11,dtype=np.float32)
    obs,rew,term,trunc,info=env.step(a)
    print(i, rew, term, info)
env.close()

import numpy as np
from env_client import make_env
env=make_env(); s,info=env.reset(seed=3); a=np.zeros(8,np.float32)
for n in range(1,env.max_steps+2):
 s,r,term,trunc,info=env.step(a)
 if n in (1,env.max_steps-1,env.max_steps,env.max_steps+1) or term or trunc:
  print(n,r,term,trunc,info)
 if term or trunc: break
env.close()

from env_client import make_env
import numpy as np, time
env = make_env(); obs,_ = env.reset(seed=0)
a = np.zeros(11, np.float32)
env.step(a)
t0=time.time()
for _ in range(100): env.step(a)
print("per-step sec", (time.time()-t0)/100)
env.close()

from env_client import make_env
import numpy as np
np.set_printoptions(precision=4, suppress=True)
e = make_env()
for seed in range(8):
 s,i=e.reset(seed=seed)
 print(seed, s.tolist(), i)
e.close()

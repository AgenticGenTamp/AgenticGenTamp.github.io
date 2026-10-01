from env_client import make_env
import numpy as np
np.set_printoptions(precision=4,suppress=True)
e=make_env()
for seed in [27,345,51,55]:
 s,_=e.reset(seed=seed)
 print(seed,'box',s[16:23],'can',s[32:39],flush=True)
e.close()

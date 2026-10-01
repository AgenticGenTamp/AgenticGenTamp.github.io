from env_client import make_env
import numpy as np
np.set_printoptions(precision=3,suppress=True)
e=make_env()
for seed in range(6):
 s,_=e.reset(seed=seed); print(seed,'base',s[16:19],'large',s[:3],'beam',s[38:45],s[51:54],'small',s[54:57],s[70:73])
e.close()

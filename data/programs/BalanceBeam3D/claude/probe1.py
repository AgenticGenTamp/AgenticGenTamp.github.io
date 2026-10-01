import numpy as np
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
env = make_env()
for s in [0,1,2]:
    obs,info = env.reset(seed=s)
    print("seed",s)
    print(" large_block", obs[0:3], "bb", obs[13:16])
    print(" robot base", obs[16:19], "arm", obs[19:26], "grip", obs[26])
    print(" seesaw", obs[38:41], "q", obs[41:45], "bb", obs[51:54])
    print(" sb1", obs[54:57], "bb", obs[67:70])
    print(" sb2", obs[70:73], "bb", obs[83:86])
    print(" info", info)
env.close()

import numpy as np
from env_client import make_env
np.set_printoptions(precision=3, suppress=True)
env = make_env()
for seed in [0,1,2]:
    obs, info = env.reset(seed=seed)
    print("seed", seed, "info", info)
    o=np.asarray(obs)
    for name,i in [("bowl",0),("drink",16),("can",32)]:
        print(" ",name, o[i:i+3], "bb", o[i+13:i+16])
    print("  cooking", o[48:55])
    print("  upper", o[57:64])
    print("  island", o[64:71])
    print("  lcorner", o[77:84], "lside", o[84:91])
    print("  robot", o[93:104])
# zero action
obs,info=env.reset(seed=0)
a=np.zeros(11,dtype=np.float32)
for t in range(5):
    obs,r,term,trunc,info=env.step(a)
    print(t, r, term, trunc, info, np.asarray(obs)[93:104])
env.close()

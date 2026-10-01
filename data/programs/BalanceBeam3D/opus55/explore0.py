from env_client import make_env
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
env = make_env()
for s in range(4):
    obs, info = env.reset(seed=s)
    print("seed", s, info)
    for name, i in [("large",0),("seesaw",38),("sb1",54),("sb2",70)]:
        print(name, obs[i:i+7], obs[i+13:i+16])
    print("robot", obs[16:27])
env.close()

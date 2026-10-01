import numpy as np
from env_client import make_env
np.set_printoptions(precision=3, suppress=True, linewidth=150)
env = make_env()
for s in [0,1]:
    obs, info = env.reset(seed=s)
    print("seed",s, info)
    for i in range(5): print("cube",i, obs[16*i:16*i+16])
    print("cook", obs[80:89]); print("upper",obs[89:96]); print("island",obs[96:109]); print("lcorner",obs[109:116]); print("lside",obs[116:125])
    print("robot",obs[125:147]); print("wiper",obs[147:163])
obs,r,te,tr,info=env.step(np.zeros(11,dtype=np.float32)); print(r,te,tr,info)
env.close()

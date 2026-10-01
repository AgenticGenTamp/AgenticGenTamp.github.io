import numpy as np
from env_client import make_env
np.set_printoptions(precision=4, suppress=True)
env = make_env()
obs, _ = env.reset(seed=0)
obs=np.asarray(obs,float)
print("start base", obs[93:96], "A", obs[0:3], "B", obs[16:19], "C", obs[32:35])
a=np.zeros(11); a[0]=-0.5
for i in range(20):
    obs,r,te,tr,info=env.step(a)
obs=np.asarray(obs,float)
print("after -x base drive")
print("base", obs[93:96], "A", obs[0:3], "B", obs[16:19], "C", obs[32:35])
print("rew",r,"term",te,"trunc",tr,"info",info)
env.close()

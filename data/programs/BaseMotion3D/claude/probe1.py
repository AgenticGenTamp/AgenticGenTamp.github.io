import numpy as np
from env_client import make_env
env = make_env()
np.set_printoptions(precision=4, suppress=True)
for s in range(3):
    obs,info = env.reset(seed=s)
    print("seed",s,obs, info)
# action effects
obs,info = env.reset(seed=0)
print("base", obs)
for i in range(11):
    e = make_env()
    o,_ = e.reset(seed=0)
    a = np.zeros(11); a[i]=0.4
    o2,r,t,tr,inf = e.step(a)
    print(i, (o2-o).round(4), r,t,tr)
    e.close()

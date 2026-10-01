import numpy as np
from env_client import make_env
def test(d):
    env=make_env(); o,_=env.reset(seed=3)
    tx,ty=o[19],o[20]
    gx,gy=tx-d, ty
    for i in range(20):
        a=np.zeros(11)
        a[0]=np.clip(gx-o[0],-0.4,0.4); a[1]=np.clip(gy-o[1],-0.4,0.4)
        o,rr,t,tr,_=env.step(a)
        if t: return True
    return False
for d in [0.0,0.001,0.005,0.01,0.02,0.03,0.04,0.05]:
    print(d, test(d))

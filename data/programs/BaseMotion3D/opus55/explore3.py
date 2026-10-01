import numpy as np
from env_client import make_env
env=make_env()
o,i=env.reset(seed=4); tgt=o[19:21].astype(float)
print(tgt)
for off in [0.3,0.2,0.15,0.1,0.05,0.02]:
    o,i=env.reset(seed=4)
    a=np.zeros(11); a[0]=0.4
    for k in range(3): o,r,te,tr,_=env.step(a)
    # now at 1.2; move to tgt-off in x
    a=np.zeros(11); a[0]=tgt[0]-off-o[0]; a[1]=tgt[1]
    o,r,te,tr,_=env.step(a)
    print(off,o[:2],te)
    # y offset
    o,i=env.reset(seed=4)
    a=np.zeros(11); a[0]=0.4
    for k in range(4): o,r,te,tr,_=env.step(a)
    a=np.zeros(11); a[0]=tgt[0]-o[0]; a[1]=tgt[1]-o[1]+off
    o,r,te,tr,_=env.step(a); print(' y',off,o[:2],te)

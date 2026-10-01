import numpy as np
from env_client import make_env
env=make_env()
np.set_printoptions(precision=3,suppress=True)
o,i=env.reset(seed=0)
a=np.zeros(11); a[2]=0.4
for t in range(4): o,*_=env.step(a)
a=np.zeros(11); a[0]=0.1
o,*_=env.step(a); print('after rot then dx', o[:3])
for s in range(5):
    o,i=env.reset(seed=s); tgt=o[19:21]; n=0
    while True:
        d=tgt-o[:2]; a=np.zeros(11); a[:2]=np.clip(d,-0.4,0.4)
        o,r,te,tr,_=env.step(a); n+=1
        if te or tr or n>30: break
    print(s,n,te,o[:3],tgt)

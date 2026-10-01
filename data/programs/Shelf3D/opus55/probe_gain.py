from env_client import make_env
from helpers import *
import sys
env=make_env()
for g in [1.0,1.5,2.0,3.0]:
    obs,info=env.reset(seed=1)
    s=rstate(obs); qt=s[3:10]+np.array([0,0.3,0,0.3,0,0.3,0])
    tr=[]
    for t in range(30):
        s=rstate(obs); e=wrap(qt-s[3:10]); tr.append(round(float(np.abs(e).max()),3))
        a=np.zeros(11,np.float32); a[3:10]=np.clip(g*e,-.1,.1); obs,*_=env.step(a)
    print(g,tr)

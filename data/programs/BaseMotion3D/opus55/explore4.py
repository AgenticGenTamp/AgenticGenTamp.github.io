import numpy as np
from env_client import make_env
env=make_env()
def test(dx,dy):
    o,i=env.reset(seed=4); tgt=o[19:21].astype(float)
    a=np.zeros(11); a[0]=0.4
    for k in range(4): o,r,te,tr,_=env.step(a)
    a=np.zeros(11); a[0]=tgt[0]-o[0]+dx; a[1]=tgt[1]-o[1]+dy
    o,r,te,tr,_=env.step(a); return te
for off in []:
    print(off,test(0,off),test(0,-off),test(off,0),test(-off,0),test(off/1.414,off/1.414))
for off in [0.03,0.04,0.045,0.05,0.055]:
    print(off,test(0,off),test(0,-off),test(off,0),test(-off,0),test(off/1.414,off/1.414))

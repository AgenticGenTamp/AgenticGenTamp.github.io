import numpy as np
from env_client import make_env
env=make_env(); o,_=env.reset(seed=0)
# base limits
for sign,idx in [(1,0),(-1,0),(1,1),(-1,1)]:
    e=make_env(); oo,_=e.reset(seed=0)
    a=np.zeros(11); a[idx]=sign*0.4
    for k in range(30):
        oo,r,t,tr,_=e.step(a)
        if t: break
    print("axis",idx,"sign",sign,"->",oo[:3].round(4),"term",t)
    e.close()
# joint limits
for j in range(3,10):
    for sign in (1,-1):
        e=make_env(); oo,_=e.reset(seed=0)
        a=np.zeros(11); a[j]=sign*0.4
        for k in range(40):
            oo,r,t,tr,_=e.step(a)
            if t: break
        print("joint",j,"sign",sign,"->",round(float(oo[j]),4))
        e.close()

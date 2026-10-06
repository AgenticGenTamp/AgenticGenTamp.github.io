import numpy as np, sys
from common import *
env=make_env()
qa=np.array([-0.61,1.49,3.39,-1.04,-0.42,-0.66,0.17])
for (m,T,vmax) in [(0.1,0.3,1.5),(0.1,0.2,1.5),(0.1,0.15,2),(0.1,0.3,3),(0.1,0.5,1.5)]:
    s,_=env.reset(seed=0, options={'object_count':1})
    errs=[]
    for i in range(30):
        base,q=robot(s)
        e=qa-q
        a=np.zeros(18); a[3:10]=np.clip(e,-m,m); a[11:18]=np.clip(e/T,-vmax,vmax)
        s,*_=env.step(a)
        errs.append(np.abs(qa-robot(s)[1]).max())
    print(m,T,vmax, np.round(errs[::3],3), np.round(qa-robot(s)[1],3), 'final', round(errs[-1],4))

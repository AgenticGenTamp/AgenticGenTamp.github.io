import numpy as np, time
from grip_util import *
t=time.time(); g=G(0); print('reset',time.time()-t)
print(g.obs.get_object_names())
c=g.cubes(); print(len(c)); print(g.bins())
for n,p in c.items(): print(n,p.round(4), isolated(c,n))
print(g.base(), g.tool())
t=time.time()
for k in range(20): g.step()
print('20 steps',time.time()-t)

import numpy as np, time
from calib_util import R
for seed in [0,1,2]:
    r=R(seed); o=r.obs
    names=o.get_object_names()
    if seed==0: print(names)
    s=o.get_object_from_name('scoop_0')
    print(seed,'scoop',r.P('scoop_0').round(4),r.Q('scoop_0').round(3),[o.get(s,f) for f in ['bb_x','bb_y','bb_z']],'base',r.base().round(3))
t=time.time()
for k in range(50): r.step()
print('step time',(time.time()-t)/50)

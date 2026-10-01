import sys, numpy as np
from env_client import make_env
out=[]
for s in range(int(sys.argv[1]),int(sys.argv[2])):
    e=make_env(); o,i=e.reset(seed=s)
    n=len([x for x in o.get_object_names() if x.startswith('cuboid')])
    m=len([x for x in o.get_object_names() if x.startswith('cupboard')])
    ys=sorted([round(float(o.data[o.get_object_from_name(x)][1]),3) for x in o.get_object_names() if x.startswith('cupboard')])
    o2,r,t,tr,_=e.step(np.zeros(11,dtype=np.float32))
    out.append((s,n,m,r,ys))
    e.close()
for x in out: print(x)

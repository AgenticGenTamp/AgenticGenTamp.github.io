import numpy as np
from env_client import make_env
import pl
env=make_env(); d=pl.Drv(env)
from collections import Counter
c=Counter()
for oc in (1,2,3,4,5,6):
    for sd in range(12):
        o=d.reset(seed=sd,oc=oc)
        ts=[]
        for n in pl.parts(o):
            f=pl.feat(o,n); ts.append('t%d'%f['triangle_type'] if 'triangle_type' in f else 'C')
        c[(oc,tuple(sorted(ts)))]+=1
        if oc<=3: print(oc,sd,ts,[np.round(pl.ppos(o,n),3).tolist() for n in pl.parts(o)],flush=True)
print(c)
# also default counts
cnt=Counter()
for sd in range(30):
    o=d.reset(seed=sd); cnt[len(pl.parts(o))]+=1
print('default count dist',cnt)
env.close()

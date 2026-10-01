import numpy as np, json
from fk import fk
d=json.load(open('calib.json')); base=np.array(d['base'])
for i,r in enumerate(d['rows']):
    q=np.array(r['q']); M=fk(q)
    Pp=np.array(r['part'][:3])-np.array([base[0],base[1],0.0])
    off=M[:3,:3].T@(Pp-M[:3,3])
    print(i,r['blk'],np.round(off,4))

import sys, numpy as np
from test_approach import make
from approach import *
env=make(); s,_=env.reset(seed=int(sys.argv[1]), options={'object_count':int(sys.argv[2])})
ap=GeneratedApproach(None,None,{}); r,rects=ap._parse(s)
print('robot',np.round(ap._q(r),2))
for n,d in sorted(rects.items()):
    c=rect_verts(d['x'],d['y'],d['th'],d['w'],d['h']).mean(0)
    print(n,np.round(c,2),round(d['w'],2),round(d['h'],2))

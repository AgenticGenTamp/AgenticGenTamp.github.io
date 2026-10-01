from probe_push_10 import grasp
from probe_push_9 import press2
from probe_push_lib import *
import sys
mode=sys.argv[1]; xos=[float(v) for v in sys.argv[2].split(',')]; res=[]
for xo in xos:
  for yf in [0.1,0.15,0.2]:
    p=P(135); ok1=grasp(p,xoff=xo,yfast=yf)
    if ok1: continue
    p.goto(gap=0.25); p.goto(y=1.2); hx=p.h()['x']
    if mode=='press': press2(p,0.2,0.2,3.499,0.015,0.12); p.goto(y=1.2); hm=p.h()['x']; ok=grasp(p)
    else: hm=hx; ok=grasp(p,clip=3.499,dy=float(mode))
    res.append((ok,round(hx,4),round(hm,4))); p.env.close()
print(mode,'ok %d/%d'%(sum(r[0] for r in res),len(res)),res,flush=True)

from probe_push_10 import grasp
from probe_push_9 import press2
from probe_push_lib import *
import sys
mode=sys.argv[1]; res=[]
for xo,yf in [(-0.0003,0.1),(0.0003,0.2),(-0.0006,0.1),(-0.0006,0.2),(0.001,0.1),(-0.0003,0.2)]:
    p=P(135); ok1=grasp(p,xoff=xo,yfast=yf)
    if ok1: res.append('first-ok'); continue
    p.goto(gap=0.25); p.goto(y=1.2); hx=p.h()['x']
    if mode=='press': press2(p,0.2,0.2,3.499,0.015,0.12); p.goto(y=1.2); hm=p.h()['x']; ok=grasp(p)
    else: hm=hx; ok=grasp(p,clip=3.499)
    res.append((ok,round(hx,4),round(hm,4))); p.env.close()
print(mode,res,flush=True)

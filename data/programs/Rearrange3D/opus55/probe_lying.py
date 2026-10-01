from probe_lib import *
import sys
seed=int(sys.argv[1]); obj=int(sys.argv[2]); pr=P(seed); pr.grip=1.0
c=pr.obs[obj:obj+3].copy(); print('obj',c,pr.obs[obj+3:obj+7])
for dx,dy in [(0,0),(0.04,0),(-0.04,0),(0,0.04),(0,-0.04),(0.07,0),(-0.07,0),(0,0.07),(0,-0.07)]:
    top=[c[0]+dx,c[1]+dy,0.62]
    pr.goto(pr.ik(top))
    ref=pr.obs[obj:obj+3].copy()
    p,why,qe=pr.line(top,[c[0]+dx,c[1]+dy,0.44],ds=0.002,qthr=0.006,watch=obj,thr=0.002)
    print((dx,dy),'contact tip z',round(pr.fk()[2],3),why,'obj moved',(pr.obs[obj:obj+3]-ref).round(3))
    pr.goto(pr.ik(top))

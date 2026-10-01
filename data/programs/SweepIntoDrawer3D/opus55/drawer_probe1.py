from drawer_util import *
import sys
r=R()
for z in [0.40,0.30,0.15]:
  for y in [-0.08]:
    q,e=r.ik([1.0,y,z],RA); r.goto(q,80); print('start',r.tool()[:3,3],e)
    x=1.0
    while x>0.4:
        x-=0.01
        q,e=r.ik([x,y,z],RA); err=r.goto(q,15)
        if err>0.02: break
    print('z',z,'y',y,'contact cmd x',x,'tool',r.tool()[:3,3],'err',err,'drawers',r.obs[103:109])
    q,e=r.ik([1.0,y,z],RA); r.goto(q,60)
print(set(r.rews), r.n)

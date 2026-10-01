import numpy as np, sys
from th import H
from approach import GeneratedApproach
from kin import fk, ik, down_R
gap=float(sys.argv[1])
h=H(478,oc=8)
ap=GeneratedApproach(None,None,{}); ap.reset(h.obs,{})
for t in range(39):
    a=ap.get_action(h.obs); h.step(a)
def go(pos,yaw):
    for k in range(40):
        q=h.q(); qt,e=ik(h.base(),q,np.asarray(pos,float),down_R(yaw),iters=100)
        d=qt-q; m=np.abs(d).max()
        if m<1e-5: return True
        a=np.zeros(11); a[3:10]=d*min(1,0.2/m); h.step(a)
        if np.allclose(h.q(),q): return False
    return False
n='obstruction7'; p=h.pose(n)[:3]; he=h.he(n); top=p[2]+he[2]; yy=np.pi/6
go((p[0],p[1],0.21),yy); go((p[0],p[1],top+0.025),yy); h.grip(-1); print('g7',h.grasped())
off=h.tool()[2,3]-p[2]
b=h.pose('target_block')[:3]; bhe=h.he('target_block')
# yaw 30 deg: obj rotates by 0 (held with same yaw at place) -> AABB same
dest=np.array([b[0]-bhe[0]-he[0]-gap, b[1], 0.075+he[2]+off+0.0015])
go((p[0],p[1],0.23),yy); go((dest[0],dest[1],0.23),yy); r=go(dest,yy); print('reached',r,h.tool()[:3,3].round(4))
h.grip(1); h.grip(1); print('released', not h.grasped(), h.pose(n)[:3].round(4), 'gapx', round(b[0]-bhe[0]-(h.pose(n)[0]+he[0]),4))
go((dest[0],dest[1],0.23),yy)
for yb in [np.pi/2, 0.0]:
    go((b[0],b[1],0.23),yb); ok=go((b[0],b[1],b[2]+bhe[2]+0.025),yb); h.grip(-1); print('block yaw',yb,'reached',ok,'grasped',h.grasped())
    if h.grasped(): break
    h.grip(1); go((b[0],b[1],0.23),yb)

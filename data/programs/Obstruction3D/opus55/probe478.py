import numpy as np, copy
from th import H
from approach import GeneratedApproach
from kin import fk, ik, down_R
h=H(478,oc=8)
ap=GeneratedApproach(None,None,{}); ap.reset(h.obs,{})
for t in range(39):
    a=ap.get_action(h.obs); h.step(a)
print(ap.phase, ap.task['name'], 'yaw', ap.task['yaw'], 'tool', h.tool()[:3,3].round(3))
n=ap.task['name']; p=h.pose(n)[:3]; he=h.he(n); top=p[2]+he[2]
q0=h.q(); y=ap.task['yaw']
def go(pos,yaw):
    for k in range(30):
        q=h.q(); qt,e=ik(h.base(),q,np.asarray(pos,float),down_R(yaw),iters=100)
        d=qt-q; m=np.abs(d).max()
        if m<1e-5: return True
        a=np.zeros(11); a[3:10]=d*min(1,0.2/m); h.step(a)
        if np.allclose(h.q(),q): return False
    return False
for dz in [0.031,0.028,0.025,0.02]:
    ok=go((p[0],p[1],top+dz),y)
    h.grip(-1); g=h.grasped()
    print('dz',dz,'reached',ok,h.tool()[:3,3].round(3),'grasped',g)
    if g: break
    h.grip(1)
for k in range(12):
    yy=k*np.pi/12
    go((p[0],p[1],0.2),yy); ok=go((p[0],p[1],top+0.025),yy)
    h.grip(-1); g=h.grasped(); print('yaw',round(np.degrees(yy)),'reached',ok,'grasped',g)
    if g: break
    h.grip(1)

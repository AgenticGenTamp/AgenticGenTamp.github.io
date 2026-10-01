import numpy as np
from th import H
from approach import GeneratedApproach
from kin import fk, ik, down_R
h=H(478,oc=8)
ap=GeneratedApproach(None,None,{}); ap.reset(h.obs,{})
for t in range(39):
    a=ap.get_action(h.obs); h.step(a)
def go(pos,yaw):
    for k in range(30):
        q=h.q(); qt,e=ik(h.base(),q,np.asarray(pos,float),down_R(yaw),iters=100)
        d=qt-q; m=np.abs(d).max()
        if m<1e-5: return True
        a=np.zeros(11); a[3:10]=d*min(1,0.2/m); h.step(a)
        if np.allclose(h.q(),q): return False
    return False
for n in ['obstruction2','obstruction5','obstruction7']:
    p=h.pose(n)[:3]; he=h.he(n); top=p[2]+he[2]; res=[]
    for k in range(0,12,1):
        yy=k*np.pi/12
        go((p[0],p[1],0.21),yy); ok=go((p[0],p[1],top+0.025),yy)
        h.grip(-1); g=h.grasped(); res.append((k*15,int(ok),int(g)))
        if g:
            go((p[0],p[1],top+0.03),yy); h.grip(1); 
        else: h.grip(1)
        go((p[0],p[1],0.21),yy)
    print(n,res)

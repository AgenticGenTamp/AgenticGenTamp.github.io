from probe_push_7 import fpts
from probe_push_lib import *
import numpy as np, sys
def press2(p,A,ARM,XR,DY,D,y0=0.53):
    p.goto(y=2.2); p.goto(x=3.0); p.goto(y=1.4); p.goto(th=-np.pi/2+A,arm=ARM,gap=0.25)
    r=p.r(); dx=XR-fpts(r)[:,0].max(); p.goto(x=r['x']+dx)
    r=p.r(); p.goto(y=r['y']+(y0-fpts(r)[:,1].min()))
    n=0; mn=9
    while True:
        r=p.r(); low=fpts(r)[:,1].min()
        if low<=0.5-D+1e-4: break
        p.step([0,max(-DY,0.5-D-low),0,0,0]); n+=1; mn=min(mn,p.h()['x'])
        if p.r()==r: break
    return n,mn
if __name__=='__main__':
    seed=int(sys.argv[1]); A,ARM,XR,DY,D=[float(v) for v in sys.argv[2:7]]
    res=[]
    for dxr in [0,-0.0003,0.0003,-0.0006,0.0006]:
        p=P(seed); h0=p.h(); n,mn=press2(p,A,ARM,XR+dxr,DY,D); p.goto(y=1.2); h1=p.h(); res.append(round(h1['x'],3)); p.env.close()
    print(sys.argv[2:],'ok %d/5'%sum(x<3.47 for x in res),res,'n',n,flush=True)

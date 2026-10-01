from probe_hook_lib import *
import numpy as np, sys, itertools
def corners(R,a,arm,gap):
    th=-np.pi/2+a; d=np.array([np.cos(th),np.sin(th)]); n=np.array([-d[1],d[0]])
    pts=[]
    c=R+arm*d
    for s in (-1,1):
        for t in (-1,1): pts.append(c+s*0.02*d+t*0.125*n)
        for u in (0,0.12):
            for t in (-0.02,0.02): pts.append(R+(arm+u)*d+(s*gap/2+t)*n)
    return np.array(pts)
def plan(hx,a,arm,gap,depth):
    th=-np.pi/2+a; d=np.array([np.cos(th),np.sin(th)])
    L=arm+0.12-(depth/2)/np.cos(a)
    P_=np.array([hx-0.025,0.5-depth/2]); R=P_-L*d
    pts=corners(R,a,arm,gap)
    return R, pts[:,0].max(), pts[:,1].min()
def run(seed,a,arm,gap,depth):
    p=P(seed); h=p.h(); R,xm,ym=plan(h['x'],a,arm,gap,depth)
    if R[0]>3.3 or xm>3.5: p.env.close(); return None
    p.goto(y=2.2); p.goto(x=3.0); p.goto(y=1.4); p.goto(th=-np.pi/2+a,arm=arm,gap=gap)
    p.goto(x=R[0]); p.goto(y=R[1]); rr=p.r(); hh=p.h()
    for k in range(20):
        p.step([0,0,0,0,-0.015])
        if p.h()['held']==1: break
    out=(rr,hh,p.r(),p.h()); p.env.close(); return out
if __name__=='__main__':
    seed=int(sys.argv[1]); arms=[float(v) for v in sys.argv[2].split(',')]
    for arm,a,gap,depth in itertools.product(arms,[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],[0.25,0.18],[0.04,0.08,0.12]):
        o=run(seed,a,arm,gap,depth)
        if o is None: continue
        rr,hh,r,h=o
        print(seed,a,arm,gap,depth,'at',rr['x'],rr['y'],'pre',hh['x'],hh['theta'],'HELD' if h['held'] else 'no',h,flush=True)

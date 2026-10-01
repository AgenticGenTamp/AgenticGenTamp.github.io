from probe_hook_lib import *
from probe_wall_5 import corners
import numpy as np, sys, itertools
def plan2(a,arm,gap,depth,xo=3.495):
    th=-np.pi/2+a; d=np.array([np.cos(th),np.sin(th)]); n=np.array([-d[1],d[0]])
    tip_out=(arm+0.12)*d+(gap/2+0.02)*n; tip_in=(arm+0.12)*d+(gap/2-0.02)*n
    Rx=min(3.3,xo-tip_out[0]); Ry=0.5-depth-tip_in[1]
    R=np.array([Rx,Ry]); pts=corners(R,a,arm,gap)
    return R,(R+tip_in),pts[:,0].max()
def run(seed,a,arm,gap,depth,dy=0.005):
    p=P(seed); h=p.h(); R,ti,xm=plan2(a,arm,gap,depth)
    p.goto(y=2.2); p.goto(x=3.0); p.goto(y=1.4); p.goto(th=-np.pi/2+a,arm=arm,gap=gap)
    p.goto(x=R[0]); 
    while p.r()['y']>R[1]+1e-3:
        y0=p.r()['y']; p.step([0,max(-dy,R[1]-y0),0,0,0])
        if p.r()['y']==y0: break
    rr=p.r(); hh=p.h()
    for k in range(20):
        p.step([0,0,0,0,-0.015])
        if p.h()['held']==1: break
    out=(ti,rr,hh,p.r(),p.h()); p.env.close(); return out
if __name__=='__main__':
    seed=int(sys.argv[1]); arms=[float(v) for v in sys.argv[2].split(',')]
    for arm,a,gap,depth in itertools.product(arms,[0.1,0.2,0.3,0.4,0.5,0.6],[0.25,0.18],[0.03,0.08]):
        ti,rr,hh,r,h=run(seed,a,arm,gap,depth)
        print(seed,a,arm,gap,depth,'tip_in %.3f %.3f'%tuple(ti),'at',rr['x'],rr['y'],'pre',hh['x'],hh['theta'],'HELD' if h['held'] else 'no',h,flush=True)

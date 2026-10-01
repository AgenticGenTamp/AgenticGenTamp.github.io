from probe_hook_lib import *
import numpy as np, sys, itertools, time
def geo(a,arm,gap):
    th=-np.pi/2+a; d=np.array([np.cos(th),np.sin(th)]); n=np.array([-d[1],d[0]])
    tip_out=(arm+0.12)*d+(gap/2+0.02)*n; tip_in=(arm+0.12)*d+(gap/2-0.02)*n
    return d,n,tip_out,tip_in
def run(seed,a,arm,gap,px,py,dy=0.005,verbose=False):
    # place outer finger inner-tip corner at hook corner + (px,py)
    p=P(seed); h=p.h(); d,n,to,ti=geo(a,arm,gap)
    R=np.array([h['x']+px,h['y']+0.5+py])-ti
    R[0]=min(R[0],3.3)
    p.goto(y=2.2); p.goto(x=3.0); p.goto(y=1.4); p.goto(th=-np.pi/2+a,arm=arm,gap=gap)
    p.goto(x=R[0]); rx=p.r()['x']
    while p.r()['y']>R[1]+1e-3:
        y0=p.r()['y']; p.step([0,max(-dy,R[1]-y0),0,0,0])
        if p.r()['y']==y0: break
    rr=p.r(); hh=p.h(); ok=False
    for k in range(25):
        p.step([0,0,0,0,-0.015])
        if p.h()['held']==1: ok=True; break
    return p,ok,R,rr,hh
if __name__=='__main__':
    seed=int(sys.argv[1])
    t=time.time()
    p,ok,R,rr,hh=run(seed,0.3,0.2,0.25,0.0,0.0)
    print(time.time()-t, ok,R,rr,hh,p.r(),p.h())

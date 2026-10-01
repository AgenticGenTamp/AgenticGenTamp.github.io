from probe_hook_lib import *
import numpy as np, time, sys
def attempt(p,a,arm,gap,rx=3.3,y0=1.3,verbose=False):
    p.goto(y=2.2); p.goto(x=3.0); p.goto(y=y0)
    p.goto(th=-np.pi/2+a,arm=arm,gap=gap)
    for k in range(200):
        r=p.r()
        if r['x']>=rx-1e-4: break
        p.step([min(0.01,rx-r['x']),0,0,0,0])
        if p.r()['x']==r['x']: break
    h0=p.h()
    for k in range(300):
        r=p.r(); p.step([0,-0.005,0,0,0]); 
        if p.r()['y']==r['y'] or p.h()!=h0: break
    rd=p.r()
    for k in range(20):
        p.step([0,0,0,0,-0.015])
        if p.h()['held']==1: break
    return rd,p.r(),p.h()
if __name__=='__main__':
    t=time.time()
    p=P(1); print(attempt(p,0,0.2,0.25)); print(time.time()-t)

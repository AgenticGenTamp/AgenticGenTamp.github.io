from probe_hook_lib import P
import numpy as np
def std_grasp(p, dy=0.005):
    h=p.h(); x=min(3.3,min(3.495,h['x']+0.015)-0.20568)
    p.goto(y=2.2); p.goto(x=3.0); p.goto(y=1.4); p.goto(th=-np.pi/2+0.2,arm=0.2,gap=0.25); p.goto(x=x)
    yt=h['y']+0.71376; n=0
    while p.r()['y']>yt+1e-4:
        y0=p.r()['y']; p.step([0,max(-(0.03 if y0>yt+0.1 else dy),yt-y0),0,0,0]); n+=1
        if p.r()['y']==y0: break
    k=0
    for k in range(1,31):
        p.step([0,0,0,0,-0.015])
        if p.h()['held']==1: break
    ok=p.h()['held']==1
    y1=p.r()['y']; hy0=p.h()['y']; p.goto(y=y1+0.3)
    lifted=p.h()['y']-hy0
    return ok, lifted

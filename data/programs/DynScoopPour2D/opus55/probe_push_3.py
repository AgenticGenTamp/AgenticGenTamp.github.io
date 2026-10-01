from probe_push_lib import *
import numpy as np
p=P(135)
h=p.h(); x=min(3.3,min(3.495,h['x']+0.015)-0.20568)
p.goto(y=2.2); p.goto(x=3.0); p.goto(y=1.4); p.goto(th=-np.pi/2+0.2,arm=0.2,gap=0.25); p.goto(x=x)
print(p.r(),p.h())
yt=h['y']+0.71376
while p.r()['y']>yt+1e-4:
    y0=p.r()['y']; p.step([0,max(-(0.03 if y0>yt+0.1 else 0.005),yt-y0),0,0,0]); print(p.r()['y'],p.h())

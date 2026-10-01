from probe_hook_lib import *
import numpy as np, sys
seed=int(sys.argv[1]); rx=float(sys.argv[2]); gap=float(sys.argv[3])
p=P(seed); h=p.h(); print('hook0',h)
p.goto(y=2.2); p.goto(th=-np.pi/2,arm=0.2,gap=gap); p.goto(x=rx); p.goto(y=1.0)
for yy in [0.6,0.5,0.45,0.42,0.40,0.38,0.36,0.34]:
    p.goto(y=yy); print('y',yy,p.r()['y'],p.h())
r=p.r()
for k in range(15):
    p.step([-0.01,0,0,0,0])
print('drag left',p.r(),p.h())

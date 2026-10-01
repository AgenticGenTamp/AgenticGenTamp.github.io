from probe_hook_lib import *
import numpy as np, sys
seed=int(sys.argv[1]); rx=float(sys.argv[2]); gap=float(sys.argv[3]); yc=float(sys.argv[4]); dx=float(sys.argv[5])
p=P(seed)
p.goto(y=2.2); p.goto(th=-np.pi/2,arm=0.2,gap=gap); p.goto(x=rx); p.goto(y=0.9)
p.goto(y=yc); print('after desc',p.r()['y'],p.h())
for k in range(40):
    p.step([dx,0,0,0,0])
    if k%5==4: print('x',p.r()['x'],p.h())

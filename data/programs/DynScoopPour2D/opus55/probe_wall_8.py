from probe_hook_lib import *
import numpy as np, sys
seed=int(sys.argv[1]); rx=float(sys.argv[2]); gap=float(sys.argv[3]); yc=float(sys.argv[4])
p=P(seed)
p.goto(y=2.2); p.goto(th=-np.pi/2,arm=0.2,gap=gap); p.goto(x=rx); p.goto(y=0.9)
p.goto(y=yc); s='y %.2f desc %s |'%(yc,p.h())
for k in range(12):
    p.step([0,0,0,0,-0.015]); h=p.h(); s+=' %.3f:%.3f,%.3f,%d'%(p.r()['finger_gap'],h['x'],h['theta'],h['held'])
print(s)

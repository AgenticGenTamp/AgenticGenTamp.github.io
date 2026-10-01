from probe_hook_lib import *
import numpy as np, sys
seed=int(sys.argv[1]); ds=[float(v) for v in sys.argv[2].split(',')]; ydes=float(sys.argv[3]) if len(sys.argv)>3 else 0.74
for d in ds:
    p=P(seed); h=p.h()
    p.goto(y=2.2); p.goto(th=-np.pi/2,arm=0.2,gap=0.25); p.goto(x=h['x']-0.026+d); p.goto(y=1.0)
    p.goto(y=h['y']+ydes)
    rd=p.r()
    for k in range(20):
        p.step([0,0,0,0,-0.015])
        if p.h()['held']==1: break
    print(seed,'d',d,'reached',rd['x'],rd['y'],'held',p.h(),p.r()['finger_gap'],flush=True); p.env.close()

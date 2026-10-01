from probe_push_lib import *
import numpy as np, sys
seed=int(sys.argv[1]); X=float(sys.argv[2]); A=float(sys.argv[3]); ARM=float(sys.argv[4]); YF=float(sys.argv[5]); DY=float(sys.argv[6]); YE=float(sys.argv[7])
p=P(seed); h0=p.h()
p.goto(y=2.2); p.goto(x=3.0); p.goto(y=1.4); p.goto(th=-np.pi/2+A,arm=ARM,gap=0.25); p.goto(x=X)
tr=[]
while p.r()['y']>YE+1e-4:
    y0=p.r()['y']; p.step([0,max(-(0.03 if y0>YF else DY),YE-y0),0,0,0]); tr.append(p.h()['x'])
    if p.r()['y']==y0: break
hend=p.h(); p.goto(y=1.2); h1=p.h()
print(sys.argv[2:],'min %.4f end %.4f lifted %.4f th %.4f'%(min(tr),hend['x'],h1['x'],h1['theta']),flush=True)

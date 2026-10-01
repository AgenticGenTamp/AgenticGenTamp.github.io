from probe_push_lib import *
import numpy as np, sys, time
# press on vertical bar top: args seed A X ARM GAP
seed=int(sys.argv[1]); A=float(sys.argv[2]); X=float(sys.argv[3]); ARM=float(sys.argv[4]); GAP=float(sys.argv[5]); DY=float(sys.argv[6]) if len(sys.argv)>6 else 0.005
t=time.time()
p=P(seed); h0=p.h()
p.goto(y=2.2); p.goto(x=3.0); p.goto(y=1.4); p.goto(th=-np.pi/2+A,arm=ARM,gap=GAP); p.goto(x=X)
tr=[]
for i in range(60):
    y0=p.r()['y']; p.step([0,-DY if y0<1.0 else -0.03,0,0,0]); h=p.h(); tr.append((round(p.r()['y'],3),h['x'],h['theta']))
    if p.r()['y']==y0: break
hmin=min(t[1] for t in tr)
print(sys.argv[1:],'minx %.4f'%hmin,'end',tr[-1],'n',len(tr),'%.1fs'%(time.time()-t))
p.goto(y=1.0); h1=p.h(); print('  after lift',h1)

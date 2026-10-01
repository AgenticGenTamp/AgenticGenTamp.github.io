from probe_push_lib import *
import numpy as np, sys
seed=int(sys.argv[1]); X=float(sys.argv[2]); ARM=float(sys.argv[3]); GAP=float(sys.argv[4]); PRESS=float(sys.argv[5]); DX=float(sys.argv[6]); N=int(sys.argv[7])
p=P(seed); h0=p.h()
p.goto(y=2.2); p.goto(x=X); p.goto(th=-np.pi/2,arm=ARM,gap=GAP); 
ytouch=0.05+ARM+0.12
p.goto(y=ytouch+0.05)
# slow descend until hook moves or robot blocked, then press
while p.r()['y']>ytouch-PRESS:
    y0=p.r()['y']; p.step([0,max(-0.005,ytouch-PRESS-y0),0,0,0])
    if p.r()['y']==y0: print('blocked at',y0); break
print('after press r',p.r(),'h',p.h())
for i in range(N):
    p.step([-DX,0,0,0,0])
    if i%5==4 or i==N-1: print(i,'r',p.r()['x'],p.r()['y'],'h',p.h())
p.goto(y=1.0)
print('h after raise',p.h(),'moved',h0['x']-p.h()['x'])
print('grasp',std_grasp(p),p.h())

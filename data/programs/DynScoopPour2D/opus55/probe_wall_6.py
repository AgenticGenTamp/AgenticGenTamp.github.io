from probe_hook_lib import *
import numpy as np, sys
seed=int(sys.argv[1]); rx=float(sys.argv[2]); gap=float(sys.argv[3]); yc=float(sys.argv[4])
p=P(seed)
p.goto(y=2.2); p.goto(th=-np.pi/2,arm=0.2,gap=gap); p.goto(x=rx); p.goto(y=0.9)
prev=None
while p.r()['y']>yc+1e-3:
    r0=p.r(); p.step([0,max(-0.01,yc-r0['y']),0,0,0]); h=p.h(); r=p.r()
    if h!=prev: print('y',r['y'],h); prev=h
    if r['y']==r0['y']: print('blocked'); break
for k in range(12):
    p.step([0,0,0,0,-0.015]); h=p.h()
    if h!=prev: print('close gap',p.r()['finger_gap'],h); prev=h

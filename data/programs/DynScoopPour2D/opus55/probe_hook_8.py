from probe_hook_lib import *
p=P(); p.goto(th=-np.pi/2); p.goto(y=1.3); p.goto(x=3.105); p.goto(y=0.74)
print('pre',p.r(),p.h())
for k in range(15):
    p.step([0,0,0,0,-0.015]); print(k,p.r()['finger_gap'],p.h())
    if p.h()['held']>0.5: break

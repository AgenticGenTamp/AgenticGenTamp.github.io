from probe_hook_lib import *
import sys
xg=float(sys.argv[1]); arm=float(sys.argv[2])
p=P(); p.goto(th=-np.pi/2); p.goto(y=1.3,arm=arm); p.goto(x=xg)
h1=p.h(); cause=None
for k in range(2000):
    r0=p.r(); p.step([0,-0.0005,0,0,0]); h=p.h(); r=p.r()
    if r['y']==r0['y']: cause='blocked';break
    if abs(h['x']-h1['x'])+abs(h['y']-h1['y'])+abs(h['theta']-h1['theta'])>3e-4: cause='hookmoved';break
print(cause,'robot',r,'hook',h,'tipY',round(r['y']-r['arm_joint'],4))

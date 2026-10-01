from probe_hook_lib import *
import itertools
def trial(seed,th,arm):
    p=P(seed); p.goto(y=1.9); p.goto(th=th,arm=arm); p.goto(x=3.4)  # as far right as possible
    h1=p.h(); 
    for k in range(3000):
        r0=p.r(); p.step([0,-0.0005,0,0,0]); h=p.h(); r=p.r()
        if r['y']==r0['y'] or abs(h['x']-h1['x'])+abs(h['theta']-h1['theta'])>3e-4: break
    p.step([0,0.001,0,0,0],2)
    p.step([0,0,0,0,-0.015],12); h2=p.h(); p.goto(y=1.3); h3=p.h()
    print(seed,'th',th,'arm',arm,'robot',p.r()['x'],round(r['y'],3),'held',h2['held'],'lifted',h3['held'],h3['y']); p.env.close()
for th,arm in itertools.product([-1.57,-1.3,-1.1,-0.9,-0.7],[0.2,0.4]): trial(1,th,arm)

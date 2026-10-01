from probe_hook_lib import *
for th in [-1.57,-1.45,-1.3]:
    p=P(1); p.goto(y=2.2); p.goto(th=th); p.goto(x=3.4); p.goto(y=1.0)
    h1=p.h()
    for k in range(2000):
        r0=p.r(); p.step([0,-0.001,0,0,0]); h=p.h(); r=p.r()
        if r['y']==r0['y'] or abs(h['x']-h1['x'])+abs(h['theta']-h1['theta'])>3e-4: break
    print('th',th,'contact',r,h)
    for k in range(10): p.step([-0.01,-0.002,0,0,0])
    print('   after drag-left',p.r(),p.h()); p.env.close()

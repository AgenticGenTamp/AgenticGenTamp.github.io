from probe_hook_lib import *
for seed in range(6):
    p=P(seed); print(seed,'robot',p.r(),'hook',p.h()); p.env.close()
def trial(dx,ry):
    p=P(); p.goto(th=-np.pi/2); p.goto(y=1.3); p.goto(x=3.1311-0.026+dx); p.goto(y=ry)
    h0=p.h(); p.step([0,0,0,0,-0.015],12); h=p.h()
    p.goto(y=1.2); h2=p.h()
    print('dx',dx,'ry',ry,'held',h['held'],'gap',p.r()['finger_gap'],'hook shift during close',round(h['x']-h0['x'],4),'lifted hook y',h2['y']); p.env.close()
for dx in [-0.03,-0.015,0.015,0.03]: trial(dx,0.74)
for ry in [0.66,0.70,0.80,0.84]: trial(0,ry)

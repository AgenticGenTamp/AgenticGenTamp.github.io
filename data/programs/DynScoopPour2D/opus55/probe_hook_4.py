from probe_hook_lib import *
def descend(xp):
    p=P(); p.goto(th=np.pi/2); p.goto(x=xp,y=0.8)
    h0=p.h()
    for k in range(800):
        y0=p.r()['y']; p.step([0,-0.001,0,0,0]); h=p.h()
        if p.r()['y']==y0 or abs(h['x']-h0['x'])+abs(h['y']-h0['y'])+abs(h['theta']-h0['theta'])>1e-4: break
    print('descend x',xp,'robot y',p.r()['y'],'hook',h, 'bottom',round(p.r()['y']-0.2,4))
    p.env.close()
for xp in [2.5,2.85,3.13,3.30]:
    descend(xp)

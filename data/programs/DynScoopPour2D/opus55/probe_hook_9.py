from probe_hook_lib import *
p=P(); p.goto(th=-np.pi/2); p.goto(y=1.3); p.goto(x=3.105); p.goto(y=0.74)
p.step([0,0,0,0,-0.015],11)
def rel():
    r=p.r(); h=p.h(); c,s=np.cos(r['theta']),np.sin(r['theta'])
    dx,dy=h['x']-r['x'],h['y']-r['y']
    return (round(c*dx+s*dy,4), round(-s*dx+c*dy,4), round((h['theta']-r['theta']+np.pi)%(2*np.pi)-np.pi,4), h['held'], r['finger_gap'])
print('grasped',p.r(),p.h(),rel())
p.step([0,0.03,0,0,0],20); print('up',p.r()['y'],p.h(),rel())
p.step([-0.03,0,0,0,0],10); print('left',p.r()['x'],p.h(),rel())
p.step([0,0,0.098,0,0],8); print('rot+',p.r()['theta'],p.h(),rel())
p.step([0,0,0,0.08,0],2); print('arm+',p.r()['arm_joint'],p.h(),rel())
p.step([0,0,0,-0.08,0],2); print('arm-',p.r()['arm_joint'],p.h(),rel())
p.step([0,0,-0.098,0,0],8); print('rot back',p.r()['theta'],p.h(),rel())
# floor blocking
for k in range(60):
    y0=p.r()['y']; p.step([0,-0.03,0,0,0])
    if p.r()['y']==y0: break
print('down blocked?',k,p.r(),p.h(),rel())
p.step([0.03,0,0,0,0],40); print('right push',p.r(),p.h(),rel())
p.step([0,0.03,0,0,0],10); p.step([-0.03,0,0,0,0],60); print('left to wall?',p.r(),p.h(),rel())
p.step([0,0,0,0,0.015],3); print('open',p.r(),p.h())
p.step([0,0.03,0,0,0],10); p.step([0]*5,30); print('after move away',p.r(),p.h())

from probe_hook_lib import *
p=P()
p.goto(th=0.0); p.step([0.03,0,0,0,0],80)
for k in range(40):
    x0=p.r()['x']; p.step([0.001,0,0,0,0]); 
    if p.r()['x']==x0: break
print('right fine (th0,arm.2)',p.r())
p.step([0,0.03,0,0,0],80)
for k in range(40):
    y0=p.r()['y']; p.step([0,0.001,0,0,0]);
    if p.r()['y']==y0: break
print('top fine',p.r())
# arm extension fine
for k in range(300):
    a0=p.r()['arm_joint']; p.step([0,0,0,0.001,0])
    if p.r()['arm_joint']==a0: break
print('arm fine',p.r())
p.step([-0.03,0,0,0,0],10); p.goto(th=np.pi/2)
for k in range(40):
    y0=p.r()['y']; p.step([0,0.001,0,0,0]);
    if p.r()['y']==y0: break
print('top fine th=pi/2',p.r())

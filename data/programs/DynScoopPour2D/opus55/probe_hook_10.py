from probe_hook_lib import *
p=P(); p.goto(th=-np.pi/2); p.goto(y=1.3); p.goto(x=3.105); p.goto(y=0.74)
p.step([0,0,0,0,-0.015],11)
p.goto(y=1.2); print('lifted',p.r(),p.h())
p.step([0,0,0,0,0.015],4); print('open mid-air',p.r()['finger_gap'],p.h())
p.step([0,0.03,0,0,0],5); p.step([0]*5,50); print('50 steps later',p.h())
# regrasp: bar top now at hook.y+0.5
h=p.h(); p.goto(x=h['x']-0.026); p.goto(y=h['y']+0.74)
p.step([0,0,0,0,-0.015],12); print('regrasp',p.r(),p.h())
# held hook vs middle wall: move left at height where hook would hit wall (wall top ~1.5)
for k in range(60):
    x0=p.r()['x']; p.step([-0.03,0,0,0,0])
    if p.r()['x']==x0: break
print('left blocked',k,p.r(),p.h())

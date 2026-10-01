from probe_hook_lib import *
p=P()
p.goto(th=np.pi); p.step([0.03,0,0,0,0],80)
for k in range(40):
    x0=p.r()['x']; p.step([0.001,0,0,0,0]); 
    if p.r()['x']==x0: break
print('right fine th=pi',p.r())
p.step([-0.2/.03*0+-0.03,0,0,0,0],10); p.goto(th=0)
for g in [0.25,0.08]:
  p.goto(gap=g); p.step([0.03,0,0,0,0],30)
  for k in range(40):
    x0=p.r()['x']; p.step([0.001,0,0,0,0]);
    if p.r()['x']==x0: break
  print('right fine th=0 gap',g,p.r())
  p.step([-0.03,0,0,0,0],10)
p.goto(arm=0.4); p.step([0.03,0,0,0,0],30)
for k in range(40):
    x0=p.r()['x']; p.step([0.001,0,0,0,0]);
    if p.r()['x']==x0: break
print('right fine th=0 arm.4',p.r())

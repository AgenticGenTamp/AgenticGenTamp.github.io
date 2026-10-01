from probe_hook_lib import *
p=P()
print(p.r())
# action semantics: dx world or local?
p.step([0.03,0,0,0,0]); print('dx',p.r())
p.step([0,0,0.098,0,0]); print('dth',p.r())
p.step([0,0,0,0.08,0]); print('darm',p.r())
p.step([0,0,0,0,-0.015]); print('dgrip',p.r())
p.step([0,0,0,0.08,0],10); print('arm max',p.r())
p.step([0,0,0,-0.08,0],10); print('arm min',p.r())
p.step([0,0,0,0,0.015],10); print('gap max',p.r())
p.step([0,0,0,0,-0.015],30); print('gap min',p.r())
# right wall
p.goto(th=0.0); p.step([0.03,0,0,0,0],80); print('right push',p.r())
p.step([0,0.03,0,0,0],80); print('up push',p.r())
p.step([0,0,0,0.08,0],10); print('arm ext at corner',p.r())

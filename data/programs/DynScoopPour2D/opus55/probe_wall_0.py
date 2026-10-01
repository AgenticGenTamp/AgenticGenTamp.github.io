from probe_hook_lib import *
import numpy as np
p=P(1)
p.goto(y=2.2); p.goto(th=-np.pi/2,arm=0.2,gap=0.25); p.goto(x=3.2); p.goto(y=1.0)
for g in [0.25,0.08]:
    p.goto(gap=g)
    for k in range(100): p.step([0.002,0,0,0,0])
    print('gap',g,p.r())
    p.goto(x=3.2)

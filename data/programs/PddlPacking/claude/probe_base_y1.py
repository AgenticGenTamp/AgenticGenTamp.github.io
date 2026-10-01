import numpy as np
from probe_base_lib import E
for y in [0.92,0.95,1.0,1.05,1.1,-0.92,-0.95,-1.0,-1.05]:
    e=E(0); e.goto_y(y)
    x,rej=e.push_x(0.05,limit=1.2)
    if rej: x,_=e.push_x(0.01,limit=1.2)
    print(f"y={y:+.2f} xmax={'%+.3f'%x if rej else 'FREE'}")
    e.close()
# approach y=1.0 from the far side: go to y=1.4, x=+0.6, then move to y=1.0
e=E(0); e.goto_y(1.4); e.goto_x(0.6)
ok=e.goto_y(1.0,stepsz=0.05)
print("from x=+0.6 side, goto y=1.0:", ok, np.round(e.r()[:3],3))
e.close()

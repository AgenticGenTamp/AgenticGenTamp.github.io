import numpy as np
from probe_base_lib import E
for y in [0.0,0.2,0.25,0.3,0.5]:
    e=E(0); e.goto_y(y)
    x,_=e.push_x(0.05,limit=1.0); x,_=e.push_x(0.01,limit=1.0); x,_=e.push_x(0.002,limit=1.0)
    print(f"y={y:+.2f} exact xmax={x:+.4f}")
    e.close()
# effect of rotation on the y=0 approach
for rot in [-1.5,-0.5,0.0,0.5,1.5,3.14]:
    e=E(0); e.goto_y(0.0)
    n=0; ok=True
    while abs(e.r()[2]-rot)>1e-6 and n<200:
        if e.move_base(dr=np.clip(rot-e.r()[2],-0.1,0.1)): ok=False; break
        n+=1
    x,rej=e.push_x(0.05,limit=1.0)
    if rej: x,_=e.push_x(0.01,limit=1.0)
    print(f"rot={rot:+.2f} (rot_ok={ok}, actual={e.r()[2]:+.2f}) xmax at y=0: {'%+.3f'%x if rej else 'FREE'}")
    e.close()

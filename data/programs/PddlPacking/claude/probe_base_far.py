import numpy as np
from probe_base_lib import E
# approach from far +x side: go to y=1.6 (clear), x=+1.5, then to target y, then push -x
for y in [0.0,0.2,0.4,0.6,0.8,0.9,1.0,-0.4,-0.8,-1.0]:
    e=E(0); e.goto_y(1.6); e.goto_x(1.5)
    if not e.goto_y(y,stepsz=0.05):
        print(f"y={y:+.2f}: could not reach y at x=1.5 (stuck {np.round(e.r()[:2],2)})"); e.close(); continue
    x,rej=e.push_x(0.05,limit=1.6,sign=-1)
    if rej: x,_=e.push_x(0.01,limit=1.6,sign=-1)
    print(f"y={y:+.2f} xmin(from +x side)={'%+.3f'%x if rej else 'FREE(<-1.6)'}")
    e.close()

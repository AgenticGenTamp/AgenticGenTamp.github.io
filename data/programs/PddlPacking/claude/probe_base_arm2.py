import numpy as np, ctrl
from probe_base_lib import E
def move_arm(e,q,maxit=80):
    for _ in range(maxit):
        d=ctrl.wrapd(np.array(q)-e.r()[3:10])
        if np.abs(d).max()<1e-5: return True
        a=np.zeros(11); a[3:10]=np.clip(d,-0.2,0.2)
        if e.step(a): return False
    return False
tuck=np.array([1.5,1.2,0.0,-2.0,0.0,-1.0,0.0])
for label,q in [("reset",None),("tuck",tuck)]:
    for y in [0.3,0.5,-0.5,0.8]:
        e=E(0)
        if q is not None: move_arm(e,q)
        e.goto_y(y)
        x,rej=e.push_x(0.05,limit=1.0)
        if rej: x,_=e.push_x(0.01,limit=1.0)
        print(f"arm={label:5s} y={y:+.2f} xmax={'%+.3f'%x if rej else 'FREE'}")
        e.close()

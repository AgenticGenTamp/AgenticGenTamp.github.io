import numpy as np, ctrl, fk
from probe_base_lib import E

e=E(0); r=e.r(); print("reset state:", np.round(r,3))
p,R,_=fk.fk_world(r[:3], r[3:10]) if hasattr(fk,'fk_world') else (None,None,None)
print("gripper world pos at reset:", np.round(p,3))
e.close()

def move_arm(e,q,maxit=80):
    for _ in range(maxit):
        d=ctrl.wrapd(np.array(q)-e.r()[3:10])
        if np.abs(d).max()<1e-5: return True
        a=np.zeros(11); a[3:10]=np.clip(d,-0.2,0.2)
        if e.step(a): return False
    return False

for label,q in [("reset", None), ("zeros", np.zeros(7)), ("tuck", np.array([1.5,1.2,0.0,-2.0,0.0,-1.0,0.0]))]:
    for y in [0.0, 1.0, -1.0]:
        e=E(0)
        if q is not None:
            ok=move_arm(e,q)
            if not ok:
                print(f"{label} y={y:+.1f}: arm move rejected, q={np.round(e.r()[3:10],2)}")
        e.goto_y(y)
        x,rej = e.push_x(0.05, limit=1.0)
        if rej: x,_ = e.push_x(0.01, limit=1.0)
        rr=e.r()
        pw,_,_ = fk.fk_world(rr[:3], rr[3:10])
        print(f"arm={label:6s} y={y:+.1f} xmax={'%+.3f'%x if rej else 'FREE'} gripper={np.round(pw,2)}")
        e.close()

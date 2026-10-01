import numpy as np
from probe_base_lib import E

for y in [-1.5, 1.5, -0.9, -0.7]:
    e = E(0)
    okx = e.goto_x(-1.0)   # already there
    oky = e.goto_y(y)
    r = e.r()
    if not oky:
        print(f"y={y:+.2f}  BLOCKED moving in y, stuck at {np.round(r[:3],3)}")
        e.close(); continue
    xmax, rej = e.push_x(0.05, limit=1.5)
    # refine with 0.01 steps
    if rej:
        xf, _ = e.push_x(0.01, limit=1.5)
    else:
        xf = xmax
    print(f"y={y:+.2f}  x_max(0.05)={xmax:+.3f}  x_max(refined 0.01)={xf:+.3f}  blocked={rej}  state={np.round(e.r()[:3],3)}")
    e.close()

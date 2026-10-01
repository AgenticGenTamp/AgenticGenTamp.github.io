import numpy as np
from probe_base_lib import E

ys = [-1.2,-0.9,-0.6,-0.4,-0.2, 0.0, 0.2, 0.4, 0.6, 0.9, 1.2]
rows = []
for y in ys:
    e = E(0)
    oky = e.goto_y(y)
    if not oky:
        rows.append((y, None, None, "y-blocked at %s" % np.round(e.r()[:2],3)))
        e.close(); continue
    x5, rej = e.push_x(0.05, limit=1.6)
    x1 = x5
    if rej:
        x1, _ = e.push_x(0.01, limit=1.6)
    rows.append((y, x5, x1, "blocked" if rej else "no block (>=1.6)"))
    print(f"y={y:+.2f}  x_max@0.05={x5:+.3f}  refined@0.01={x1:+.3f}  {rows[-1][3]}")
    e.close()
print()
print("y      xmax")
for y,x5,x1,note in rows:
    print(f"{y:+.2f}  {('%+.3f'%x1) if x1 is not None else 'n/a':>7}  {note}")

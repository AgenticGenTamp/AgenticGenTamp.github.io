import numpy as np
from probe_base_lib import E

ys = [round(-1.4+0.1*i,2) for i in range(29)]
res={}
for y in ys:
    e = E(0)
    if not e.goto_y(y):
        res[y]=None; print(f"y={y:+.2f} y-blocked"); e.close(); continue
    x5, rej = e.push_x(0.05, limit=1.0)
    x1 = x5
    if rej: x1,_ = e.push_x(0.01, limit=1.0)
    res[y] = x1 if rej else None
    print(f"y={y:+.2f}  xmax={'%+.3f'%x1 if rej else 'FREE(>1.0)'}")
    e.close()
np.save("probe_base_map.npy", np.array([[y, (v if v is not None else np.nan)] for y,v in res.items()]))

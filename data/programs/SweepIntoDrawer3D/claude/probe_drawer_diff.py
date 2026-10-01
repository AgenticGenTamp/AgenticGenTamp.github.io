import numpy as np
from env_client import make_env
from pngtool import readpng
env=make_env(); obs,_=env.reset(seed=0)
base=env.render_state(state=obs.tolist(), label="dd_base")
A=readpng(base).astype(int)
for j in range(6):
    o=obs.copy(); o[103+j]=0.30
    p=env.render_state(state=o.tolist(), label="dd_%d"%j)
    B=readpng(p).astype(int)
    d=(np.abs(B-A).max(2)>25)
    ys,xs=np.nonzero(d)
    if len(xs)==0: print(j,'no change'); continue
    # dark pixels that changed
    dk=d & ((A.max(2)<70)|(B.max(2)<70))
    dy,dx=np.nonzero(dk)
    print(f'j={j} n={len(xs):5d} u[{xs.min()},{xs.max()}] v[{ys.min()},{ys.max()}] cen=({xs.mean():.1f},{ys.mean():.1f}) '
          f'| dark n={len(dx)} u[{dx.min() if len(dx) else -1},{dx.max() if len(dx) else -1}] v[{dy.min() if len(dy) else -1},{dy.max() if len(dy) else -1}]')
env.close()

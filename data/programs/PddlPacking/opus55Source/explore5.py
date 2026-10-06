import numpy as np, time
from simcore import *
s = Sim(4)
st = s.get_state()
q0 = np.array(st[:10], float)
rng=np.random.default_rng(0)
for by in [0.0]:
  for bx in [-0.45]:
    print('base',bx,by)
    t=time.time()
    for ty in np.arange(0.6,-0.61,-0.1):
        row=''
        for tx in np.arange(-0.3,0.31,0.05):
            ok=False
            for yaw in [0, np.pi/2, -np.pi/2, np.pi]:
                q=q0.copy(); q[0]=bx; q[1]=by
                if s.ik((tx,ty,0.81), top_quat(yaw), q, range(3,10)) is not None: ok=True; break
            row+='Y' if ok else '.'
        print('%5.2f'%ty,row)
    print(time.time()-t)

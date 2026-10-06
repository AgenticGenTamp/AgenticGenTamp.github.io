import numpy as np, time
from simcore import *
s = Sim(4)
st = s.get_state()
q0 = np.array(st[:10], float)
print('q0', q0)
# reachability with base fixed at x=bx,y=0
for bx in [-0.9, -0.8, -0.7, -0.6]:
    q = q0.copy(); q[0] = bx
    s.set_state(st)
    s.set_conf(q)
    with s.env.client():
        coll = s.env._in_collision()
    row = []
    for tx in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
        t=time.time()
        r = s.ik((tx, 0.2, 0.81), top_quat(0.0), q, range(3, 10))
        row.append('Y' if r is not None else '.')
    print(bx, 'coll', coll, ''.join(row), time.time()-t)

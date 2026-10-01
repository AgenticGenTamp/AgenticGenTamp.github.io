import numpy as np
from calib_util import *
r=R()
print('init fk', r.fk()[0], r.fk()[1].round(3))
r.goto_base([-0.14,0.0,0.0]); print('base', r.base())
tgt=np.array([0.2,-0.2,0.65])
q,err=r.ik_world(tgt); print('ik err',err, q.round(3))
r.goto_q(q)
print('q actual-target', (r.q()-q).round(4), 'fk', r.fk()[0])
# lower progressively and look at tracking
for z in np.arange(0.60,0.38,-0.01):
    q,err=r.ik_world(np.array([0.2,-0.2,z])); r.goto_q(q,settle=8)
    print(round(z,3), 'err', round(err,5), 'track', np.abs(r.q()-q).max().round(4), 'fkz', r.fk()[0].round(4))

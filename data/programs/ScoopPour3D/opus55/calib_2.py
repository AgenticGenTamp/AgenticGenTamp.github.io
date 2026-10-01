import numpy as np, sys
from calib_util import *
r=R()
cn = sys.argv[1] if len(sys.argv)>1 else 'cube_12'
r.goto_base([-0.14,0.0,0.0])
r.gripper(1.0,20); print('closed grip', r.grip())
r.gripper(0.0,20); print('open grip', r.grip())
c=r.P(cn); print(cn, c)
names=[n for n in r.obs.get_object_names() if n.startswith('cube')]
P0={n:r.P(n) for n in names}
q,_=r.ik_world(c+[0,0,0.15]); r.goto_q(q)
for dz in [0.10,0.06,0.04,0.02,0.0,-0.01,-0.02]:
    q,_=r.ik_world(c+[0,0,dz]); r.goto_q(q,settle=8)
    print('dz',dz,'track',np.abs(r.q()-q).max().round(4),'fk',r.fk()[0].round(4))
r.gripper(1.0,25); print('grip after close', r.grip())
for n in names:
    d=r.P(n)-P0[n]
    if np.linalg.norm(d)>0.002: print('moved',n,d.round(4))
q,_=r.ik_world(c+[0,0,0.15]); r.goto_q(q,settle=10)
print('after lift', cn, r.P(cn), 'fk', r.fk()[0].round(4), 'grip', r.grip())
print('rew sum', sum(r.rew), 'nonzero', [x for x in r.rew if x!=0][:5])

import numpy as np, sys
from calib_util import *
T=-0.224
r=R()
r.goto_base([-0.14,0.0,0.0])
g=float(sys.argv[1]); r.gripper(g,10)
def depth(xy, zmin=0.42):
    q,_=r.ik_world(np.array([xy[0],xy[1],0.56]),tool=T); r.goto_q(q)
    for z in np.arange(0.54,zmin-1e-9,-0.01):
        q,_=r.ik_world(np.array([xy[0],xy[1],z]),tool=T); r.goto_q(q,settle=3)
    r.goto_q(q,settle=8)
    fz=r.fk(tool=T)[0]
    q,_=r.ik_world(np.array([xy[0],xy[1],0.56]),tool=T); r.goto_q(q)
    return fz
for y in [-0.3,-0.2,-0.1,0.0]:
    print(y, [round(depth((x,y))[2],4) for x in [0.1,0.15,0.2,0.25]], flush=True)
print('bin', r.P('bin_yellow_0').round(4), 'scoop', r.P('scoop_0').round(4))

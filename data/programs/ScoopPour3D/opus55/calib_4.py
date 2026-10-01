import numpy as np, sys
from calib_util import *
import kin
T=-0.224
r=R()
r.goto_base([-0.14,0.0,0.0])
def touch(xy, g, z0=0.62, zmin=0.40, Rw=RD):
    r.gripper(g,10)
    q,_=r.ik_world(np.array([xy[0],xy[1],z0]),Rw,tool=T); r.goto_q(q)
    last=None; base0=r.fk(tool=T)[0][2]-z0
    for z in np.arange(z0,zmin,-0.005):
        q,_=r.ik_world(np.array([xy[0],xy[1],z]),Rw,tool=T); r.goto_q(q,settle=6)
        fz=r.fk(tool=T)[0][2]
        if fz-z-base0>0.004:
            break
        last=fz
    q,_=r.ik_world(np.array([xy[0],xy[1],z0]),Rw,tool=T); r.goto_q(q)
    return last, base0
for g in [1.0,0.0]:
    print('island touch g=',g,'contact fk(T) z', touch((0.2,-0.2),g))
print('bin', r.P('bin_yellow_0'))

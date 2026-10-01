import numpy as np, sys
from calib_util import *
import kin
T=-0.224
r=R()
cn='cube_12'
r.goto_base([-0.14,0.0,0.0])
c=r.P(cn)
q,_=r.ik_world(c+[0,0,0.15]); r.goto_q(q)
q,_=r.ik_world(c+[0,0,0.02]); r.goto_q(q,settle=8)
q,_=r.ik_world(c+[0,0,-0.01]); r.goto_q(q,settle=8)
r.gripper(1.0,25)
q,_=r.ik_world(c+[0,0,0.15]); r.goto_q(q,settle=10)
print('held', r.P(cn))
q,_=r.ik_world(np.array([0.2,-0.2,0.49]),tool=T); r.goto_q(q,settle=10)
r.gripper(0.0,25)
q,_=r.ik_world(np.array([0.2,-0.2,0.58]),tool=T); r.goto_q(q,settle=10)
cp=r.P(cn); print('placed cube at', cp.round(4), 'quat', r.Q(cn).round(3))
def probe(xy, zt=0.4525):
    q,_=r.ik_world(np.array([xy[0],xy[1],0.56]),tool=T); r.goto_q(q)
    for z in [0.52,0.50,0.49,0.48,0.47,0.46,zt]:
        q,_=r.ik_world(np.array([xy[0],xy[1],z]),tool=T); r.goto_q(q,settle=6)
    fz=r.fk(tool=T)[0][2]
    q,_=r.ik_world(np.array([xy[0],xy[1],0.56]),tool=T); r.goto_q(q)
    return fz
for ax in [0,1]:
    for d in np.arange(-0.07,0.0701,0.005):
        xy=cp[:2].copy(); xy[ax]+=d
        fz=probe(xy)
        print('axis',ax,'d=%.3f'%d,'fz=%.4f'%fz, 'BLOCK' if fz>0.4525+0.008 else '', 'cube',r.P(cn)[:2].round(4), flush=True)
        if np.linalg.norm(r.P(cn)[:2]-cp[:2])>0.003: cp=r.P(cn); print('cube moved -> recentre')

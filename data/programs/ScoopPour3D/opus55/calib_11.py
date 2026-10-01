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
q,_=r.ik_world(np.array([0.2,-0.2,0.475]),tool=T); r.goto_q(q,settle=10)
r.gripper(0.0,25)
q,_=r.ik_world(np.array([0.2,-0.2,0.58]),tool=T); r.goto_q(q,settle=10)
print('placed', r.P(cn).round(4), flush=True)
def trial(xy):
    r.gripper(0.0,15)
    q,_=r.ik_world(np.array([xy[0],xy[1],0.56]),tool=T); r.goto_q(q)
    for z in [0.50,0.48,0.46,0.445]:
        q,_=r.ik_world(np.array([xy[0],xy[1],z]),tool=T); r.goto_q(q,settle=5)
    fz=r.fk(tool=T)[0][2]
    p0=r.P(cn)
    r.gripper(1.0,20); p1=r.P(cn)
    q,_=r.ik_world(np.array([xy[0],xy[1],0.56]),tool=T); r.goto_q(q,settle=5)
    p2=r.P(cn); lifted=p2[2]>0.5
    q,_=r.ik_world(np.array([xy[0],xy[1],0.475]),tool=T); r.goto_q(q,settle=5)
    r.gripper(0.0,15)
    q,_=r.ik_world(np.array([xy[0],xy[1],0.56]),tool=T); r.goto_q(q,settle=5)
    return fz,p0,p1,lifted
for ax in [1,0]:
    for d in [0.0,0.01,0.02,0.03,0.035,0.04,0.045,0.05,0.055,0.06,-0.03,-0.04,-0.045,-0.05]:
        c=r.P(cn)
        if c[2]>0.47 or np.linalg.norm(c[:2]-[0.19,-0.19])>0.06:
            print('cube lost',c); sys.exit()
        xy=c[:2].copy(); xy[ax]+=d
        fz,p0,p1,l=trial(xy)
        print('ax',ax,'d=%.3f'%d,'fz=%.4f'%fz,'disp_on_descend',(p0-c)[:2].round(4),'disp_close',(p1-p0)[:2].round(4),'lifted',l,flush=True)

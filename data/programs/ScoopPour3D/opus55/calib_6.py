import numpy as np, sys
from calib_util import *
import kin
T=-0.2245
r=R()
cn='cube_0'
r.goto_base([-0.14,0.0,0.0])
c=r.P(cn)
tgt=np.array([0.4,-0.1,0.65])
def rep(tag):
    q,_=r.ik_world(tgt,tool=T); r.goto_q(q,settle=20)
    print(tag,'qi-q',(r.qi-r.q()).round(4),'fk',r.fk(tool=T)[0].round(4),'cube',r.P(cn).round(4))
rep('no cube, open')
r.gripper(1.0,20); rep('no cube, closed')
r.gripper(0.0,20)
q,_=r.ik_world(c+[0,0,0.1],tool=T); r.goto_q(q)
q,_=r.ik_world(c+[0,0,0.0],tool=T); r.goto_q(q,settle=8)
print('at grasp: fk',r.fk(tool=T)[0].round(4),'cube',c.round(4))
r.gripper(1.0,25)
rep('holding')
r.gripper(0.0,20); rep('released')
print('bin',r.P('bin_yellow_0'))

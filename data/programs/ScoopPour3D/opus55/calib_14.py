import numpy as np, sys
from calib_util import *
T=-0.224
r=R()
r.goto_base([-0.14,0.0,0.0])
b0=r.P('bin_yellow_0'); print('bin0',b0.round(4))
wy=float(sys.argv[1]) if len(sys.argv)>1 else -0.05
zg=float(sys.argv[2]) if len(sys.argv)>2 else 0.505
def go(p,settle=5):
    q,_=r.ik_world(np.array(p),tool=T); r.goto_q(q,settle=settle)
r.gripper(0.0,10)
go([0.5,wy,0.6])
for z in np.arange(0.58,zg-1e-9,-0.01): go([0.5,wy,z],3)
go([0.5,wy,zg],8)
print('at grasp fk',r.fk(tool=T)[0].round(4),'bin',r.P('bin_yellow_0').round(4))
r.gripper(1.0,25); print('closed; grip',r.grip(),'bin',r.P('bin_yellow_0').round(4))
cubes=[n for n in r.obs.get_object_names() if n.startswith('cube')]
for dz in [0.02,0.05,0.1,0.15]:
    go([0.5,wy,zg+dz],10)
    b=r.P('bin_yellow_0')
    print('lift dz',dz,'fk z',r.fk(tool=T)[0][2].round(4),'bin',b.round(4),'quat',r.Q('bin_yellow_0').round(3),'qi-q',np.abs(r.qi-r.q()).max().round(3))
for k in range(20): r.step(dq=np.zeros(7))
print('after 20 steps hold: bin',r.P('bin_yellow_0').round(4),'cube z range',min(r.P(c)[2] for c in cubes).round(3),max(r.P(c)[2] for c in cubes).round(3))
print('rewards uniq',set(r.rew))

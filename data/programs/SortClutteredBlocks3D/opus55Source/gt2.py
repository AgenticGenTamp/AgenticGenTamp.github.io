import sys, numpy as np, mujoco
exec(open('grasptest.py').read().split("run([ox,0,0.6],8")[0])
def pads():
    out=[]
    for n in ['robot_right_pad1','robot_right_pad2','robot_left_pad1','robot_left_pad2']:
        out.append(d.geom_xpos[m.geom(n).id].round(3))
    return out
z=float(sys.argv[4])
run([ox,0,z],10,0.62); print('open',pads())
run([ox,0,z],4,1.0); print('closed',pads())

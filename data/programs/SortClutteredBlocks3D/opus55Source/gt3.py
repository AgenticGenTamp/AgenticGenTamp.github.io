import sys, numpy as np, mujoco
exec(open('grasptest.py').read().split("run([ox,0,0.6],8")[0])
def pads():
    return np.array([d.geom_xpos[m.geom(n).id] for n in ['robot_right_pad1','robot_right_pad2','robot_left_pad1','robot_left_pad2']]).round(3).tolist()
def run2(pos,n,grip):
    global s
    ap.grip=grip; ap._set_arm_target(base,pos,yaw)
    for i in range(n):
        s,*_=env.step(ap._act(*ap._robot(s)[:2], ap._vel))
        b,arm,_=ap._robot(s); ee=ap._ee(b,arm); c=ap._pose(s,'cube1')
        print('ee',ee.round(3),'c',c[:3].round(3),pads())
        for k in range(d.ncon):
            g1,g2=d.contact[k].geom1,d.contact[k].geom2
            n1,n2=m.geom(g1).name or m.body(m.geom_bodyid[g1]).name, m.geom(g2).name or m.body(m.geom_bodyid[g2]).name
            if 'robot' in n1+n2: print('   con',n1,n2)
run2([0,0,0.6],8,0.62)
run2([0,0,0.41+dz+0.06],3,0.62)
run2([0,0,0.41+dz],3,0.62)
run2([0,0,0.41+dz],3,1.0)

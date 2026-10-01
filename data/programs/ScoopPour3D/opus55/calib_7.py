import numpy as np, sys
exec(open('calib_4.py').read().split("def touch")[0])
def probe(xy,g,z0=0.6,zmin=0.45,Rw=RD):
    r.gripper(g,10)
    q,_=r.ik_world(np.array([xy[0],xy[1],z0]),Rw,tool=T); r.goto_q(q)
    for z in np.arange(z0,zmin,-0.01):
        q,_=r.ik_world(np.array([xy[0],xy[1],z]),Rw,tool=T); r.goto_q(q,settle=6)
        print(round(z,3), 'fk',r.fk(tool=T)[0].round(4),'qi-q',np.abs(r.qi-r.q()).max().round(4),'base',r.base().round(3))
    q,_=r.ik_world(np.array([xy[0],xy[1],z0]),Rw,tool=T); r.goto_q(q)
probe((0.5,-0.12),0.0)

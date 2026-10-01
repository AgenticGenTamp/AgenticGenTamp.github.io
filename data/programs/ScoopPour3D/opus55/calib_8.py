import numpy as np, sys
exec(open('calib_4.py').read().split("def touch")[0])
def probe(xy,g,z0=0.6,zmin=0.43,Rw=RD):
    r.gripper(g,10)
    q,_=r.ik_world(np.array([xy[0],xy[1],z0]),Rw,tool=T); r.goto_q(q)
    e0=np.abs(r.qi-r.q()).max(); res=None
    for z in np.arange(z0,zmin,-0.005):
        q,_=r.ik_world(np.array([xy[0],xy[1],z]),Rw,tool=T); r.goto_q(q,settle=6)
        if np.abs(r.qi-r.q()).max()-e0>0.006: res=r.fk(tool=T)[0][2]; break
    q,_=r.ik_world(np.array([xy[0],xy[1],z0]),Rw,tool=T); r.goto_q(q)
    return res
for xy in [(0.2,-0.2),(0.5,0.0),(0.5,0.02),(0.5,-0.02),(0.2,-0.05),(0.35,-0.4),(0.24,-0.2)]:
    print(xy,'open',probe(xy,0.0),'closed',probe(xy,1.0), flush=True)

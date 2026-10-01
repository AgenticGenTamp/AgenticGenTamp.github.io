import numpy as np, sys, pickle
from calib_util import *
import kin
def rotx(t): c,s=np.cos(t),np.sin(t); return np.array([[1,0,0],[0,c,-s],[0,s,c.real]])
def roty(t): c,s=np.cos(t),np.sin(t); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
r=R()
cn = sys.argv[1] if len(sys.argv)>1 else 'cube_12'
r.goto_base([-0.14,0.0,0.0])
c=r.P(cn)
q,_=r.ik_world(c+[0,0,0.15]); r.goto_q(q)
q,_=r.ik_world(c+[0,0,0.02]); r.goto_q(q,settle=8)
q,_=r.ik_world(c+[0,0,-0.01]); r.goto_q(q,settle=8)
r.gripper(1.0,25)
q,_=r.ik_world(c+[0,0,0.15]); r.goto_q(q,settle=10)
data=[]
rng=np.random.default_rng(0)
for bt in [0.0,0.35,-0.35]:
    r.goto_base([-0.14,0.0,bt],steps=60)
    for k in range(7):
        pw=np.array([rng.uniform(0.25,0.5),rng.uniform(-0.25,0.15),rng.uniform(0.6,0.8)])
        Rw=kin.rotz(rng.uniform(-0.6,0.6))@rotx(rng.uniform(-0.4,0.4))@roty(rng.uniform(-0.4,0.4))@RD
        q,err=r.ik_world(pw,Rw)
        if err>1e-3: continue
        r.goto_q(q,settle=15)
        rec=(r.base().copy(), r.q().copy(), r.P(cn).copy(), r.Q(cn).copy())
        data.append(rec)
        print(k, 'base',rec[0].round(3),'cube',rec[2].round(4),'fk',r.fk()[0].round(4), 'vel?')
pickle.dump(data,open('calib_data_%s.pkl'%cn,'wb'))

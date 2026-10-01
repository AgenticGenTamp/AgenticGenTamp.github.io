import sys; sys.path.insert(0,'.')
import numpy as np, math
from approach import *
from kin import ik_down, Q0, fk_full
b=np.array([0.031,0.323,0.781,1.628])
ap=GeneratedApproach.__new__(GeneratedApproach); ap.bad_cfgs=[]; ap.margin=0.01
w=World({'block0':b})
for base in [(-0.47,0.22,0.0),(-0.57,0.14,0.0),(-0.47,0.0,0.0)]:
    q,ok=ik_down(base,np.array([b[0],b[1],b[2]+GRASP_DZ]),Q0,yaw=b[3])
    pts,_,_,tool,R=fk_full(base,q)
    print(base, ok, np.round(q,2), np.round(tool,3), np.round(R[:,0],3))
    c=ap._ik_cfg(base,np.array([b[0],b[1],b[2]+GRASP_DZ]),Q0,yaw=b[3])
    print(" ikcfg", c is not None)
    if c is not None:
        for k in range(4):
            cc=c.copy(); cc[9]=wrap(cc[9]+k*math.pi/2)
            print("  k",k, ap._config_ok(cc,w,exclude='block0',closed=False,margin=0.0))

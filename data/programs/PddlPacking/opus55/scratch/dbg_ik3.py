import sys; sys.path.insert(0,'.')
import numpy as np, math
from approach import *
from kin import ik_down, Q0, fk_full
b=np.array([float(v) for v in sys.argv[1:5]])
bases=eval(sys.argv[5])
ap=GeneratedApproach.__new__(GeneratedApproach); ap.bad_cfgs=[]; ap.margin=0.01
w=World({'block0':b})
for base in bases:
    print(base, "base_ok", base_ok(base))
    c=ap._ik_cfg(base,np.array([b[0],b[1],b[2]+GRASP_DZ]),Q0,yaw=b[3])
    print(" ikcfg", None if c is None else np.round(c[3:],2))
    if c is not None:
        for k in range(4):
            cc=c.copy(); cc[9]=wrap(cc[9]+k*math.pi/2)
            ok=ap._config_ok(cc,w,exclude='block0',closed=False,margin=0.0)
            print("  k",k, ok)

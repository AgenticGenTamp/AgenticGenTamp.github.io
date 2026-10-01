import numpy as np
from ik2 import ik_multi
from ik import ik
from ctrl2 import tip_to_fk, RFWD, RDOWN, JLO, JHI
from fk import fk
np.set_printoptions(precision=3,suppress=True,linewidth=200)
q0=np.array([0.0,-0.3491,3.1416,-2.5482,-0.0,-0.8727,1.5708])
# roll RFWD about tool z (local x) to get fingers horizontal
def rollx(R,a):
    c,s=np.cos(a),np.sin(a)
    Rx=np.array([[1,0,0],[0,c,-s],[0,s,c]])
    return Rx@R
LIM=np.array([3.1,2.41,3.1,2.66,3.1,2.23,3.1])
for name,R in [("RFWD",RFWD),("RFWD_roll90",rollx(RFWD,np.pi/2)),("RDOWN",RDOWN)]:
    print("=== ",name)
    for bx in [1.50,1.60,1.70]:
        for wz in [0.20,0.30,0.40]:
            for wx in [0.90,0.95]:
                lx=bx-wx
                p=tip_to_fk([lx,0.0,wz],R)
                q,c=ik_multi(p,R,q0,n=40,seed=1)
                if q is None: print(f"  bx{bx} z{wz} x{wx} lx{lx:.2f}: NONE"); continue
                marg=(LIM-np.abs(q)).min()
                print(f"  bx{bx} z{wz} x{wx} lx{lx:.2f}: travel{c:.2f} limmarg{marg:.2f} q{np.round(q,2)}")

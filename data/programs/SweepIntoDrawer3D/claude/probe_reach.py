import numpy as np
from ik2 import ik_multi
from ctrl2 import RFWD,RDOWN,tip_to_fk
q0=np.array([0.0,-0.3491,3.1416,-2.5482,0,-0.8727,1.5708])
for basex in [1.242,1.45,1.6]:
    print("=== base_x",basex)
    for z in [0.20,0.30,0.40,0.46]:
        row=[]
        for wx in [1.15,1.05,0.95,0.85,0.75,0.65,0.55]:
            lx=basex-wx
            p=tip_to_fk([lx,0.0,z],RFWD)
            q,c=ik_multi(p,RFWD,q0,n=10)
            row.append(f"{wx}:{'OK' if q is not None else '--'}")
        print(f" z={z} fwd  ",row)
    for z in [0.46]:
        row=[]
        for wx in [1.05,0.95,0.85,0.75,0.65,0.55,0.45]:
            lx=basex-wx
            p=tip_to_fk([lx,0.0,z],RDOWN)
            q,c=ik_multi(p,RDOWN,q0,n=10)
            row.append(f"{wx}:{'OK' if q is not None else '--'}")
        print(f" z={z} down ",row)

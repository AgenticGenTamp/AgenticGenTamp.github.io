import numpy as np
exec(open("scratch/common.py").read())
P=np.pi
q0=np.array([0.6772,-0.3431,1.2,-1.4669,1.2422,-1.9544,2.2225])
cfgs={"swung":np.r_[2.0,q0[1:]], "swung_up":np.r_[2.0,-0.5,q0[2:]], "back":np.r_[2.2,-0.3,0,-0.3,0,-0.3,0]}
for n,q in cfgs.items():
    print(n, pts((0,0,0),q).round(2)[2:].tolist())
    try:
        print("  -x yaw0 front",round(bis([(-1,0,0)],lambda v:(v,0,0),-1,-0.3,q=q)+0.3,4),
              " -y yawpi/2 front", round(bis([(-1,-1.5,P/2),(0.2,-1.5,P/2)],lambda v:(0.2,v,P/2),-1.5,-0.6,q=q)+0.6,4),
              " -x yaw-pi/2 left", round(bis([(-1,0,-P/2)],lambda v:(v,0,-P/2),-1.5,-0.3,q=q)+0.3,4))
    except AssertionError as e: print("  fail",e)
